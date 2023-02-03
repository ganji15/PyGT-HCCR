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
        self.ffn = GeoConvBlock(gcn, D, O, **kwargs)

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

    def info(self):
        nParams = sum([p.nelement() for p in self.parameters()])
        mSize = nParams * 4.0 / 1024 / 1024
        print("%-6s #param: %d size: %.4fMB\r\n" % (self.__class__.__name__, nParams, mSize))
        # print(self)


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


