from networks.module import *
from networks.utils import *
from torch_geometric.nn import GMMConv, SplineConv
import copy


CONFIG_PyGT = {'Ti': [{'B':2, 'D':32, 'N':2}, {'B':2, 'D':48, 'N':4}, {'B':2, 'D':72, 'N':6}, {'B':2, 'D':120, 'N':8}],
          'S': [{'B':2, 'D':48, 'N':2}, {'B':2, 'D':72, 'N':4}, {'B':2, 'D':108, 'N':6}, {'B':2, 'D':160, 'N':8}],
          'M': [{'B':2, 'D':64, 'N':2}, {'B':2, 'D':96, 'N':4}, {'B':3, 'D':144, 'N':6}, {'B':2, 'D':200, 'N':8}],
          'L': [{'B':2, 'D':72, 'N':2}, {'B':3, 'D':108, 'N':4}, {'B':3, 'D':168, 'N':6}, {'B':2, 'D':256, 'N':8}]}

CONFIG_IsGT = { 'M': [{'B': 9, 'D':144, 'N': 4}],
                'L': [{'B': 10, 'D':172, 'N': 4}]}


GCNkwargs = {'gcn': GMMConv, 'kernel_size': 3, 'dim': 2, 'separate_gaussians': False}
transform = T.Cartesian
se_reduction = 8


def parseGCNkwargs(in_kwargs):
    kwargs = copy.deepcopy(in_kwargs)
    gcn = kwargs.pop('gcn')
    kwargs.pop('kernel_size')
    kwargs.pop('dim')
    return gcn, kwargs


class GraphTransformerBlock(nn.Module):
    def __init__(self, D, N, O, gcn_kwargs=GCNkwargs, attn_drop=0.2):
        super(GraphTransformerBlock, self).__init__()
        self.attn = GraphAttention(D, n_heads=N, drop_path=attn_drop)
        kwargs = copy.deepcopy(gcn_kwargs)
        gcn = kwargs.pop('gcn')
        self.ffn = GeoConvBlock(gcn, D, O, se_reduction=0, **kwargs)

    def forward(self, data):
        data = self.attn(data)
        data = self.ffn(data)
        return data


class GraphTransformerBase(nn.Module):
    def __init__(self, cfg, in_features=6, last_fc=160, gcn_kwargs=GCNkwargs, num_classes=3755, online=1, downsample=True):
        super(GraphTransformerBase, self).__init__()
        if online > 2:
            featlayers = [PositionFeatLayer(transform=T.Cartesian(cat=False), init_x=online % 2)]
        elif online > 1:
            featlayers = [BuildKnnEdge(12), OfflineFeatLayer(transform=transform(cat=False))]
        elif online > 0:
            featlayers = [SeqFeatLayer(transform=T.Cartesian(cat=False))]
        else:
            featlayers = [OfflineFeatLayer(transform=transform(cat=False))]

        gcn, kwargs = parseGCNkwargs(gcn_kwargs)
        self.head = nn.Sequential(*featlayers,
            GeoConvLayer(gcn, in_features, cfg[0]['D'], kernel_size=5, dim=2, act=nn.PReLU, **kwargs),
        )

        features = []
        for i, block_cfg in enumerate(cfg):
            for j in range(block_cfg['B']):
                O = block_cfg['D'] if j < block_cfg['B'] - 1 else cfg[min(i + 1, len(cfg) - 1)]['D']
                features.append(GraphTransformerBlock(block_cfg['D'], block_cfg['N'], O, gcn_kwargs=gcn_kwargs))
            if i < len(cfg) - 1:
                if downsample:
                    features.append(GraclusMaxPool())
            else:
                features.append(GraphGlobalPool('mean'))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Linear(cfg[-1]['D'], last_fc),
            nn.PReLU(),
            nn.Dropout(0.2),
            CosLayer(last_fc, num_classes)
        )

    def forward(self, data):
        data = self.head(data)
        x = self.features(data)
        x = x.view(-1, self.classifier[0].weight.size(1))
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

    def calc_flops(self, data:Batch):
        num_samples = data.y.shape[0]
        flops = 0
        for layer in self.head:
            if isinstance(layer, GeoConvLayer):
                flops += FlopCalculater._flop_GConv(data.x.shape[0], data.edge_index.shape[1],
                                          layer.conv.in_channels,  layer.conv.out_channels)
            data = layer(data)

        for block in self.features:
            if isinstance(block, GraphTransformerBlock):
                for layer in [block.attn]:
                    if isinstance(layer, GraphAttention):
                        _, mask = to_dense_batch(data.x, data.batch, 0)
                        seq_lens = mask.int().sum(dim=-1).detach().cpu().numpy()
                        flops += FlopCalculater._flop_GraphAttn(layer.in_dim, layer.n_heads, seq_lens)
                    data = layer(data)

                for layer in [block.ffn]:
                    if isinstance(layer, GeoConvBlock):
                        extra_flop = FlopCalculater._flop_GConv(data.x.shape[0], data.edge_index.shape[1],
                                                      layer.conv1.in_channels, layer.conv1.out_channels)\
                                    + FlopCalculater._flop_GConv(data.x.shape[0], data.edge_index.shape[1],
                                                      layer.conv2.in_channels, layer.conv2.out_channels)\
                                    + FlopCalculater._flop_Linear(layer.conv1.in_channels, layer.conv2.out_channels, data.x.shape[0])
                        flops += extra_flop
                    data = layer(data)
            else:
                data = block(data)

        # print(data.size())
        x = data.view(-1, self.classifier[0].weight.size(1))
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                flops += FlopCalculater._flop_Linear(layer.in_features, layer.out_features) * num_samples
            x = layer(x)
        return flops * 1.0 / 1e9

    def info(self):
        nParams = sum([p.nelement() for p in self.parameters()])
        mSize = nParams * 4.0 / 1024 / 1024
        print("%-6s #param: %d size: %.4fMB\r\n" % (self.__class__.__name__, nParams, mSize))
        # print(self)


class FlopCalculater():
    @staticmethod
    def _flop_Conv2D(H, W, in_cnl, out_cnl, K, N=1):
        # N x Cout x H x W x  (Cin x Kw x Kh + bias)
         return 2 * H * W * (in_cnl * K**2 + 1) * out_cnl * N
        # return H * W * (in_cnl * K**2 + 1) * out_cnl

    @staticmethod
    def _flop_Linear(in_features, out_features, N=1,  seq_len=1):
        return (2 * in_features - 1) * out_features * N * seq_len
        # return in_features * out_features * seq_len

    @staticmethod
    def _flop_GraphAttn(D, num_heads, seq_lens):
        flops = 0
        # print(seq_lens)
        # input project
        for seq_len in seq_lens:
            flops += seq_len * (2 * D - 1) * D * 3

            # attention heads: scale, matmul, softmax, matmul
            head_dim = D // num_heads
            head_flops = seq_len * head_dim + \
                         seq_len * seq_len * (2 * head_dim - 1) + \
                         seq_len * seq_len + \
                         seq_len * seq_len * (2 * seq_len - 1)
            flops += num_heads * head_flops

            # output project
            flops += seq_len * (2 * D - 1) * D
        return flops

    @staticmethod
    def _flop_SplineConv(Nv, Ne, in_cnl, out_cnl):
        return Ne * 9 * (3 * in_cnl * out_cnl + 14) + (Ne + Nv) * out_cnl

    @staticmethod
    def _flop_Poly(Nv, Ne, in_cnl, out_cnl):
        return (Ne + Nv) * out_cnl + Ne * in_cnl * out_cnl * (6 * 2 + 1)

    @staticmethod
    def _flop_QmaPoly(Nv, Ne, in_cnl, out_cnl, Head=3):
        return (Ne + Nv) * out_cnl + (Ne * out_cnl * (6 * 2 + 1) + in_cnl * out_cnl * Nv) * Head

    @staticmethod
    def _flop_QmaGMM(Nv, Ne, in_cnl, out_cnl, Head=3):
        return (Ne + Nv) * out_cnl + (Ne * out_cnl * 12 + in_cnl * out_cnl * Nv) * Head

    @staticmethod
    def _flop_GConv(*args, **kwargs):
        return FlopCalculater._flop_QmaGMM(*args, **kwargs)


class PyramidGraphTransformerBase(GraphTransformerBase):
    def __init__(self, *args, **kwargs):
        super(PyramidGraphTransformerBase, self).__init__(downsample=True, *args, **kwargs)


class PyGT_T(PyramidGraphTransformerBase):
    def __init__(self, in_features=7, num_classes=3755, online=True):
        super(PyGT_T, self).__init__(in_features=in_features, gcn_kwargs=GCNkwargs, num_classes=num_classes, online=online,
                                      cfg=CONFIG_PyGT['Ti'])


class PyGT_S(PyramidGraphTransformerBase):
    def __init__(self, in_features=7, num_classes=3755, online=True):
        super(PyGT_S, self).__init__(in_features=in_features, gcn_kwargs=GCNkwargs, num_classes=num_classes, online=online,
                                      cfg=CONFIG_PyGT['S'])


class PyGT_M(PyramidGraphTransformerBase):
    def __init__(self, in_features=7, num_classes=3755, online=True):
        super(PyGT_M, self).__init__(in_features=in_features, gcn_kwargs=GCNkwargs, num_classes=num_classes, online=online,
                                      cfg=CONFIG_PyGT['M'])


class PyGT_L(PyramidGraphTransformerBase):
    def __init__(self, in_features=7, num_classes=3755, online=True):
        super(PyGT_L, self).__init__(in_features=in_features, gcn_kwargs=GCNkwargs, num_classes=num_classes, online=online,
                                      cfg=CONFIG_PyGT['L'])


class PyGT_Abl(PyramidGraphTransformerBase):
    def __init__(self, in_features=3, num_classes=3755, online=3):
        super(PyGT_Abl, self).__init__(in_features=in_features, gcn_kwargs=GCNkwargs, num_classes=num_classes, online=online,
                                       cfg=CONFIG_PyGT['M'])

#=============================================================================
class IsotropicGraphTransformerBase(GraphTransformerBase):
    def __init__(self, *args, **kwargs):
        super(IsotropicGraphTransformerBase, self).__init__(downsample=False, *args, **kwargs)


class IsGT_M(IsotropicGraphTransformerBase):
    def __init__(self, in_features=7, num_classes=3755, online=True):
        super(IsGT_M, self).__init__(in_features=in_features, gcn_kwargs=GCNkwargs, num_classes=num_classes, online=online,
                                      cfg=CONFIG_IsGT['M'])


class IsGT_L(IsotropicGraphTransformerBase):
    def __init__(self, in_features=7, num_classes=3755, online=True):
        super(IsGT_L, self).__init__(in_features=in_features, gcn_kwargs=GCNkwargs, num_classes=num_classes, online=online,
                                      cfg=CONFIG_IsGT['L'])


# for MnistSkeletons
class LeNet_GCNet(nn.Module):
    def __init__(self, num_classes, in_features, *args, **kwargs):
        super(LeNet_GCNet, self).__init__()

        gcn = GMMConv
        transform = T.Cartesian
        kwargs = {'kernel_size': 5, 'dim': 2}

        self.features = nn.Sequential(
            OfflineFeatLayer(transform=transform(cat=False)),

            GeoConvLayer(gcn, in_features, 32, **kwargs),
            GraphAttention(32, 8, drop_path=0.2),
            GraclusMaxPool(transform=transform(cat=False)),

            GeoConvLayer(gcn, 32, 64, **kwargs),
            GraphAttention(64, 16, drop_path=0.2),
            GraclusMaxPool(transform=transform(cat=False)),

            GraphGlobalPool('mean')
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            torch.nn.PReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, data):
        x = self.features(data)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)


