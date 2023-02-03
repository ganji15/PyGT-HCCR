import math
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import knn_graph
from torch_scatter import scatter_mean

import torch_geometric.transforms as T
from torch_geometric.utils import degree
from torch_geometric.nn import (graclus, voxel_grid, max_pool,
                                global_mean_pool, global_max_pool, GlobalAttention,
                                global_sort_pool)
from torch_geometric.data import Batch
from torch_geometric.utils import to_undirected, add_remaining_self_loops, to_dense_batch


class OfflineFeatLayer(nn.Module):
    def __init__(self, spatial=True, color=True, transform=None):
        super(OfflineFeatLayer, self).__init__()
        self.transform=transform
        self.spatial = spatial
        self.color = color

    def forward(self, data:Batch):
        if self.spatial and self.color:
            data.x = torch.cat([data.pos, data.x], dim=1)
        elif self.spatial and not self.color:
            data.x = data.pos
        elif not self.spatial and not self.color:
            data.x = torch.ones_like(data.x).to(data.x.device)
        else:
            pass

        data.edge_index = to_undirected(data.edge_index)
        data.edge_index, _ = add_remaining_self_loops(data.edge_index, num_nodes=data.x.size(0))
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __repr__(self):
        return '{}(spatial={}, color={}, transform={})'.format(
            self.__class__.__name__,
            self.spatial,
            self.color,
            self.transform.__class__.__name__
        )


class SeqFeatExtractor(MessagePassing):
    def __init__(self):
        super(SeqFeatExtractor, self).__init__(flow="source_to_target", aggr='mean')

    def message(self, x_i, x_j):
        feats = []
        diff_coord = x_j - x_i
        diff_norm = torch.norm(diff_coord, p=2, dim=-1).view(-1, 1) + 1e-6
        diff_direct = diff_coord / diff_norm
        feats.append(diff_direct)

        out = torch.cat([diff_coord, diff_direct], dim=-1)
        return out

    def forward(self, x, edge_index):
        out = self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x[:, :2])
        return out


class PositionFeatLayer(nn.Module):
    def __init__(self, transform=None, init_x=False):
        super(PositionFeatLayer, self).__init__()
        self.transform=transform
        self.init_x = init_x

    def forward(self, data:Batch):
        data.x = torch.ones((data.pos.size(0), 1)) if data.x is None else data.x
        data.x = torch.cat([data.pos, data.x], dim=-1)

        data.edge_index = to_undirected(data.edge_index)
        data.edge_index, _ = add_remaining_self_loops(data.edge_index, num_nodes=data.x.size(0))
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __repr__(self):
        return '{}(transform={}, init_x={})'.format(
            self.__class__.__name__,
            self.transform.__class__.__name__,
            self.init_x
        )


class SeqFeatLayer(nn.Module):
    def __init__(self, spatial=True, temporal=True, transform=None, set_x_None=False):
        super(SeqFeatLayer, self).__init__()
        self.transform=transform
        self.spatial = spatial
        self.temporal = temporal
        self.feat_extractor = SeqFeatExtractor() if self.temporal else None
        self.set_x_None = set_x_None

    def forward(self, data:Batch):
        data.x = None if self.set_x_None else data.x
        if self.spatial and self.temporal:
            out = self.feat_extractor(data.pos, data.edge_index)
            if data.x is None:
                data.x = torch.cat([out, data.pos], dim=1)
            else:
                data.x = torch.cat([out, data.pos, data.x], dim=-1)
        elif self.spatial and not self.temporal:
            if data.x is None:
                data.x = data.pos
            elif data.x.size(1) == 1:
                data.x = torch.cat([data.pos, data.x], dim=-1)
        elif not self.spatial and self.temporal:
            if data.x is None or (data.x.size(1) == 2):
                data.x = self.feat_extractor(data.pos, data.edge_index)
        else: # elif not self.spatial and not self.temporal:
            data.x = torch.ones_like(data.pos).to(data.pos.device)
            # if data.x is None:
            #     data.x = torch.ones_like(data.pos).to(data.pos.device)
            # else:
            #     data.x = torch.cat([data.pos, data.x], dim=-1)

        data.edge_index = to_undirected(data.edge_index)
        data.edge_index, _ = add_remaining_self_loops(data.edge_index, num_nodes=data.x.size(0))
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __repr__(self):
        return '{}(spatial={}, temproal={}, transform={})'.format(
            self.__class__.__name__,
            self.spatial,
            self.temporal,
            self.transform.__class__.__name__
        )


class LinearLayer(nn.Module):
    def __init__(self, in_size, fc_size, res=False, **kwargs):
        super(LinearLayer, self).__init__()
        self.nn = nn.Linear(in_features=in_size, out_features=fc_size, **kwargs)
        self.bn = nn.BatchNorm1d(fc_size)
        self.act = nn.PReLU()
        self.is_res = res

    def forward(self, data):
        out = self.act(self.bn(self.nn(data.x)))
        if self.is_res and self.nn.in_features == self.nn.out_features:
            data.x += out
        else:
            data.x = out

        return data


class ResMLPLayer(nn.Module):
    def __init__(self, in_size, fc_size, **kwargs):
        super(ResMLPLayer, self).__init__()
        self.nn1 = nn.Linear(in_features=in_size, out_features=in_size, **kwargs)
        self.bn1 = nn.BatchNorm1d(in_size)
        self.act1 = nn.PReLU()
        self.nn2 = nn.Linear(in_features=in_size, out_features=fc_size, **kwargs)
        self.bn2 = nn.BatchNorm1d(fc_size)
        self.act2 = nn.PReLU()
        if in_size != fc_size:
            self.res = nn.Linear(in_size, fc_size, bias=False)
        else:
            self.res = nn.Identity()

    def forward(self, data):
        out = self.act1(self.bn1(self.nn1(data.x)))
        out = self.bn2(self.nn2(out))
        data.x = self.act2((self.res(data.x) + out) * math.sqrt(0.5))
        return data


class ProjectLinearLayer(nn.Module):
    def __init__(self, in_size, fc_size, res=False, **kwargs):
        super(ProjectLinearLayer, self).__init__()
        self.nn = nn.Linear(in_features=in_size, out_features=fc_size, bias=False)

    def forward(self, data):
        data.x = self.nn(data.x)
        return data


def get_gcn_nn(in_cnl, out_cnl, hiddin):
    return nn.Sequential(nn.Linear(2, hiddin),
                         nn.PReLU(),
                         nn.Linear(hiddin, in_cnl * out_cnl))


class GraphSELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(GraphSELayer, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, batch_idx, num_graphs):
        x_mean = scatter_mean(x, batch_idx, dim=0, dim_size=num_graphs)
        x_attn_score = self.attn(x_mean)
        x_atten = x * x_attn_score[batch_idx]
        return x_atten


class GraphJACLayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(GraphJACLayer, self).__init__()
        self.glb = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
        )

        self.local = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
        )

        self.attn = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x, batch_idx, num_graphs):
        x_glb = scatter_mean(self.glb(x), batch_idx, dim=0, dim_size=num_graphs)[batch_idx]
        x_atten = self.attn(self.local(x) + x_glb)
        return x_atten


class GeoConvLayer(nn.Module):
    def __init__(self, gcn, in_cnl, out_cnl, act=nn.PReLU, with_res=False, **kwargs):
        super(GeoConvLayer, self).__init__()
        self.conv = gcn(in_cnl, out_cnl, **kwargs)
        self.bn = nn.BatchNorm1d(out_cnl)
        self.act = None
        self.act = act()
        if with_res:
            self.res = nn.Sequential(
                nn.Linear(in_cnl, out_cnl, bias=False),
                nn.BatchNorm1d(out_cnl)
            )
        else:
            self.res = None

    def forward(self, data):
        res = 0 if self.res is None else self.res(data.x)
        data.x = self.conv(data.x, data.edge_index, data.edge_attr)
        data.x = self.bn(data.x)
        if self.act:
            data.x = self.act(data.x)
        data.x += res
        return data


class GeoConvBlock1(torch.nn.Module):
    def __init__(self, gcn, in_features, out_features, dropout=0.2,
                 act=nn.PReLU, se_rd=8, **kwargs):
        super(GeoConvBlock1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = gcn(in_features, out_features, **kwargs)
        self.bn1 = torch.nn.BatchNorm1d(out_features)
        self.relu1 = act()

        self.dropout = nn.Dropout(p=dropout)

        self.conv2 = gcn(out_features, out_features, **kwargs)
        self.bn2 = torch.nn.BatchNorm1d(out_features)
        self.relu2 = act()

        self.se = GraphJACLayer(out_features, se_rd) if se_rd > 0 else None
        self.res = nn.Sequential(nn.Linear(in_features, out_features, bias=False),
                                 nn.BatchNorm1d(out_features))

    def forward(self, data:Batch):
        res = self.res(data.x)
        data.x = self.relu1(self.bn1(self.conv1(data.x, data.edge_index, data.edge_attr)))
        data.x = self.dropout(data.x)
        data.x = self.bn2(self.conv2(data.x, data.edge_index, data.edge_attr))
        if self.se is not None:
            data.x = self.se(data.x, data.batch, data.num_graphs)
        data.x = self.relu2((data.x + res) * math.sqrt(0.5))
        return data


class SElayer(torch.nn.Module):
    def __init__(self, channel, reduction=16):
        super(SElayer, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.PReLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, data):
        x_avg = scatter_mean(data.x, data.batch, dim=0, dim_size=data.num_graphs)
        se_batch = self.fc(x_avg)[data.batch]
        data.x = data.x * se_batch
        return data


class GeoConvBlock(torch.nn.Module):
    def __init__(self, gcn, in_features, out_features, with_res=True, dropout=0.2,
                 act=nn.PReLU, weight_hidden=9, se_reduction=0, **kwargs):
        super(GeoConvBlock, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = gcn(in_features, out_features, root_weight=with_res, **kwargs)
        self.bn1 = torch.nn.BatchNorm1d(out_features)
        # self.bn1 = torch.nn.LayerNorm(out_features)
        self.relu1 = act()

        self.dropout = nn.Dropout(p=dropout)
        self.conv2 = gcn(out_features, out_features, **kwargs)
        self.bn2 = torch.nn.BatchNorm1d(out_features)
        self.relu2 = act()

        self.se_reduction = se_reduction
        if se_reduction > 1:
            self.se = SElayer(out_features, se_reduction)

        self.res = nn.Sequential(nn.Linear(in_features, out_features, bias=False),
                                 nn.BatchNorm1d(out_features)
                                 ) if with_res else None

    def forward(self, data):
        if self.res is not None:
            res = self.res(data.x)
            data.x = self.relu1(self.bn1(self.conv1(data.x, data.edge_index, data.edge_attr)))
            data.x = self.dropout(data.x)
            data.x = self.bn2(self.conv2(data.x, data.edge_index, data.edge_attr))
            if self.se_reduction > 1:
                data = self.se(data)
            data.x = self.relu2((data.x + res) * math.sqrt(0.5))
        else:
            data.x = self.relu1(self.bn1(self.conv1(data.x, data.edge_index, data.edge_attr)))
            data.x = self.dropout(data.x)
            data.x = self.relu2(self.bn2(self.conv2(data.x, data.edge_index, data.edge_attr)))
        return data


class FirstGeoConvBlock(nn.Module):
    def __init__(self, gcn, in_features, out_features, depth=2, act=nn.PReLU, **args):
        super(FirstGeoConvBlock, self).__init__()
        self.conv1 = gcn(in_features, out_features, **args)
        self.bn1 = torch.nn.BatchNorm1d(out_features)
        self.relu1 = act()

        self.conv2 = gcn(out_features, out_features, **args)
        self.bn2 = torch.nn.BatchNorm1d(out_features)
        self.relu2 = act()

    def forward(self, data):
        data.x = self.relu1(self.bn1(self.conv1(data.x, data.edge_index, data.edge_attr)))
        data.x = self.relu2(self.bn2(self.conv2(data.x, data.edge_index, data.edge_attr)))
        return data


class GeoConvBlock2(torch.nn.Module):
    def __init__(self, gcn, in_features, out_features, depth=2, **args):
        super(GeoConvBlock2, self).__init__()
        self.depth = depth

        if self.depth > 1:
            first_layer = [GeoConvLayer(gcn, in_features, out_features, act='relu', **args)]
            mid_layers = [GeoConvLayer(gcn, out_features, out_features, act='relu', **args) for _ in range(depth - 2)]
            last_layer = [GeoConvLayer(gcn, out_features, out_features, act='', **args)]
            self.main = nn.Sequential(*(first_layer + mid_layers + last_layer))
            self.relu = nn.PReLU()
            self.res = nn.Sequential(nn.Linear(in_features, out_features, bias=False),
                                     nn.BatchNorm1d(out_features))
        else:
            self.main = GeoConvLayer(gcn, in_features, out_features, act='relu', **args)

    def forward(self, data):
        if self.depth > 1:
            res = self.res(data.x)
            data = self.main(data)
            data.x = self.relu((data.x + res) * math.sqrt(0.5))
        else:
            data = self.main(data)
        return data


class CosLayer(nn.Module):
    def __init__(self, in_size, out_size, s=23.0):
        super(CosLayer, self).__init__()
        self.in_size = in_size
        self.out_size = out_size

        self.W = nn.Parameter(torch.randn(out_size, in_size))
        self.W.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

        self.s = nn.Parameter(torch.randn(1,)) if s is None else s

    def forward(self, input):
        cosine = F.linear(F.normalize(input), F.normalize(self.W))
        output = cosine * self.s
        return output

    def __repr__(self):
        return self.__class__.__name__ +  '(in_size={}, out_size={}, s={})'.format(
                    self.in_size, self.out_size,
                    'learn' if isinstance(self.s, nn.Parameter) else self.s)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class GraphAttention(torch.nn.Module):
    def __init__(self, in_dim, n_heads=2, drop_path=0.):
        super().__init__()
        self.in_dim = in_dim
        self.n_heads = n_heads
        self.attn = Attention(in_dim, n_heads)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.ln = nn.LayerNorm(in_dim)
        self.attn_drop = drop_path

    def forward(self, data):
        x, mask = to_dense_batch(data.x, data.batch, 0)

        attn, attn_score = self.attn(self.ln(x), mask)
        x = x + self.drop_path(attn)

        data.x = x.masked_select(mask.unsqueeze(dim=-1)).view(-1, attn.size(-1))
        return data

    def __repr__(self):
        return 'GraphAttention(D={}, heads={}, drop={})'.format(self.in_dim, self.n_heads, self.attn_drop)


class Attention(nn.Module):
    def __init__(self, D, heads=4):
        super().__init__()
        self.D = D
        self.heads = heads

        assert (D % heads == 0), "Embedding size should be divisble by number of heads"
        self.head_dim = D // heads

        self.qkv = nn.Linear(self.D, self.D * 3, bias=False)
        self.H = nn.Linear(self.D, self.D)

    def forward(self, x, mask=None):
        bz, x_len = x.size(0), x.size(1)
        Q, K, V = self.qkv(x).chunk(3, -1)
        Q = Q.reshape(bz, x_len, self.heads, self.head_dim)
        K = K.reshape(bz, x_len, self.heads, self.head_dim)
        V = V.reshape(bz, x_len, self.heads, self.head_dim)

        scores = torch.einsum('bqhd,bkhd->bhqk', [Q, K]) # bz, heads, len, len
        if mask is not None:
            true_mask = torch.einsum('bq,bk->bqk', [mask, mask]).unsqueeze(1)
            scores = scores.masked_fill(true_mask == 0, -1e5)
        attn = torch.softmax(scores / np.sqrt(self.D), -1)
        attn_out = torch.einsum('bhqv,bvhd->bqhd', [attn, V])
        output = self.H(attn_out.reshape(bz, x_len, self.D))

        return output, attn


class TransformLayer(nn.Module):
    def __init__(self, transform):
        super(TransformLayer, self).__init__()
        self.transform = transform

    def forward(self, data):
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                               self.transform.__class__.__name__ if self.transform else 'None')


class BuildKnnEdge(nn.Module):
    def __init__(self, knn=20):
        super(BuildKnnEdge, self).__init__()
        self.knn = knn

    def forward(self, data):
        data.edge_index = knn_graph(data.pos, self.knn, batch=data.batch).detach()
        return data




def normalized_cut(edge_index, edge_attr, num_nodes: Optional[int] = None):
    row, col = edge_index[0], edge_index[1]
    deg = 1. / (degree(col, num_nodes, edge_attr.dtype) + 1e-8)
    deg = deg[row] + deg[col]
    cut = edge_attr * deg
    return cut


def normalized_cut_2d(edge_index, pos):
    row, col = edge_index
    edge_attr = torch.norm(pos[row] - pos[col], p=2, dim=1)
    return normalized_cut(edge_index, edge_attr, num_nodes=pos.size(0))


class GraclusMaxPool(nn.Module):
    def __init__(self, transform=T.Cartesian(cat=False)):
        super(GraclusMaxPool, self).__init__()
        self.transfrom = transform

    def forward(self, data):
        weight = normalized_cut_2d(data.edge_index, data.pos)
        cluster = graclus(data.edge_index, weight, data.x.size(0))
        data.edge_attr = None
        data = max_pool(cluster, data, transform=self.transfrom)
        data.edge_index, data.edge_attr =\
            add_remaining_self_loops(data.edge_index, data.edge_attr)
        # data.edge_index = to_undirected(data.edge_index)
        # data.edge_index, _ = add_remaining_self_loops( num_nodes=data.x.size(0))
        return data

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,
                               self.transfrom.__class__.__name__)


class VoxelMaxPool(nn.Module):
    def __init__(self, size=3, start=None, end=None, transform=None):
        super(VoxelMaxPool, self).__init__()
        self.size = [size, size]
        self.start = start
        self.end = end
        self.transform = transform

    def forward(self, data):
        cluster = voxel_grid(data.pos, size=self.size, start=self.start, end=self.end)
        print(cluster)
        data.edge_attr = None
        data = max_pool(cluster, data, transform=self.transform)
        return data

    def __repr__(self):
        return 'VoxelMaxPool(pool_size={})'.format(self.size)


class GraphGlobalPool(nn.Module):
    def __init__(self, mode='max', **args):
        super(GraphGlobalPool, self).__init__()
        self.mode = mode
        if mode == 'max':
            self.pool = global_max_pool

        if mode == 'mean':
            self.pool = global_mean_pool

        if mode == 'attn':
            self.pool = GlobalAttention(**args)

        if mode == 'sort':
            self.pool = global_sort_pool

        self.args = args

    def forward(self, data):
        x = self.pool(data.x, data.batch, **self.args)
        return x

    def __repr__(self):
        return 'GraphGlobalPool(mode={})'.format(self.mode)
