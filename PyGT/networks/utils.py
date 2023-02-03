import torch
from torch_geometric.utils import add_remaining_self_loops


class AddSelfLoops(object):
    def __call__(self, data):
        data.edge_index, _ = add_remaining_self_loops(data.edge_index, num_nodes=data.x.size(0))
        return data


class StorePosition(object):
    def __call__(self, data):
        data.pos = data.x
        return data


class PositionOffset(object):
    def __init__(self, offset_x, offset_y):
        self.offset_x = offset_x
        self.offset_y = offset_y

    def __call__(self, data):
        data.pos[:, 0] -= self.offset_x
        data.pos[:, 1] -= self.offset_y
        return data


class OnesDataX(object):
    def __call__(self, data):
        data.x = torch.ones_like(data.x).to(data.x.device)
        return data


class ScalePos(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, data):
        data.pos = data.pos * self.scale
        return data


class MinusMin(object):
    def __init__(self, mean=0.):
        self.mean = mean

    def __call__(self, data):
        data.x = data.x - self.mean
        return data


class CoorNorm(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        data.pos = data.pos * 2 / self.size - 1
        return data


def _info(model, ret_str=False, detail=True):
    nParams = sum([p.nelement() for p in model.parameters()])
    mSize = nParams * 4.0 / 1024 / 1024
    res = "*%s #Params: %d  #Stor.: %.4fMB\r\n" \
          % (model.__class__.__name__, nParams, mSize)
    if detail:
        res += str(model)
    if ret_str:
        return res
    print(res)
