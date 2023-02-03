import os
from scipy.io import loadmat

from traj_process.traj_normalize import normalize_trajectory, normalize_char_traj
from traj_process.traj_graph import *

from scipy.spatial import Delaunay
import matplotlib.pyplot as plt

import glob

import torch
from datasets import *
from datasets import AllDatasets as Datasets
# from torch_geometric.data import DataLoader
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Cartesian
from torch_geometric.data import Data, Batch

import pickle


def load_data(fx, fy, fz, flb):
    flb = loadmat(flb)['label'][:, 0]
    fx = loadmat(fx)['X']
    fy = loadmat(fy)['Y']
    fz = loadmat(fz)['Z']
    # return np.array(fx), np.array(fy), np.array(fz), np.array(mlb)
    for x, y, z, lb in zip(fx, fy, fz, flb):
        idx = np.nonzero(x)
        traj = np.stack([x[idx], y[idx]]).transpose()
        yield traj, lb


def try_digits():
    x, y, z, lb = 'data\X.mat', 'data\Y.mat', 'data\Z.mat', 'data\label.mat'
    for traj, lb in load_data(x, y, z, lb):
        if lb not in [4, 8]:
            continue

        traj = normalize_trajectory(traj, _correct_slope=False)
        print(lb)
        plt.subplot(121)
        plt.plot(traj[:, 0], traj[:, 1], '-o')
        plt.subplot(122)
        tri = Delaunay(traj)
        plt.triplot(traj[:, 0], traj[:, 1], tri.simplices)
        plt.plot(traj[:, 0], traj[:, 1], 'o')
        plt.show()

        nodes, edges = construct_graph(traj)
        print('len(traj): ', len(traj))
        print('len(nodes): ', len(nodes))
        # print(edges)
        plot_graph(nodes, edges)
        plt.show()

    print('over')


def try_chars():
    files = glob.glob('samples/*.txt')
    for fn in files:
        with open(fn, 'r') as f:
            f.readline()
            orj_traj = np.loadtxt(f)
            traj = normalize_char_traj(orj_traj, resample_step=0.4)
            traj = traj[::3]
            plt.subplot(121)
            plt.plot(traj[:, 0], traj[:, 1], '-*')

            # plt.subplot(122)
            # traj = normalize_char_traj(orj_traj, resample_step=0.4, merge_thres=0.12)
            # plt.plot(traj[:, 0], traj[:, 1], '-*')

            nodes, edges = construct_graph(traj, merge_thres=0.07)
            print('len(traj): ', len(traj))
            print('len(nodes): ', len(nodes))
            info = ', '.join('%d [%.2f, %.2f]'%(i, pt[0], pt[1]) for i, pt in enumerate(nodes))
            print(info)
            edges = sorted(edges, key=lambda x: (x[0], x[1]))
            print(edges)
            plt.subplot(122)
            plot_graph(nodes, edges, label_size=50, node_size=100, font_size=10)
            plt.show()


def try_dataloader():
    # train_dataset = IsoCharDataset(root='./data/', train=True)
    # test_dataset = IsoCharDataset(root='./data/', train=False)
    train_dataset = SuperPixelCharDataset(root='./data/', train=True)
    test_dataset = SuperPixelCharDataset(root='./data/', train=False)
    print(len(train_dataset))
    print(len(test_dataset))


def try_char_dataloader():
    from torch_geometric.utils import to_undirected, add_remaining_self_loops, to_dense_batch
    # from networks.module import GraclusMaxPool
    from offline_to_graph.construct_graph import plot_graph_attention
    from torch_geometric.transforms import Compose, RandomRotate, RandomShear
    # test_dataset = AirCharDataset(root='./data/', train=False)
    # test_dataset = UnipenChar(root='./data/', set_name='1c', train=False)
    # test_dataset = IsoCharDataset(root='./data/', train=False)
    # test_dataset = OfflineCharDataset(root='./data/', train=False)
    test_dataset = GenOfflineCharDataset(root='./data/', train=False)
    # test_dataset = TestIcdarGraphDataset(root='./data/', set_name='shuf')
    # test_dataset = GridMNIST(root='./data/', train=False)
    # test_dataset = SuperPixelCharDataset(root='./data/', train=False)
    # test_dataset = OfflineCharDataset(root='./data/', train=True)
    # test_dataset = MnistSkeletonsRandom(root='./data/', train=False)
    # test_dataset = RandomAirCharDataset(root='./data/', train=False)
    # test_dataset = EMnistSTkeletons(root='./data/', set_name='balanced', train=False)
    # test_dataset = MnistSkeletonsRandom(root='./data/', train=False)
    # test_dataset = MnistSkeletons(root='./data/', train=False)
    # print(len(train_dataset))
    # print(len(test_dataset))
    # test_dataset = GridMNIST(root='./data/', train=False)
    # test_dataset = MNISTSuperpixels(root='./data/', train=False)
    print(len(test_dataset))
    alphabet = test_dataset._load_alphabet()
    # transform = RandomRotate(40)
    # transform = Compose([
    #     RandomShear(0.25),
    #     RandomRotate(30)
    # ])
    # transform = None
    # pool = GraclusMaxPool()
    for data in test_dataset:
        print('is_batch ', isinstance(data, Batch))
        # print(data)
        print(alphabet[data.y.item()])
        # if data.y.item() != 3545:
        #     continue
        # print(data.pos)
        # print(data.pos.size())
        pts = data.pos.numpy()
        # print(data.x)
        if data.x is None:
            data.x = torch.zeros((data.pos.size(0),))
        plot_graph_attention(pts, data.edge_index.transpose(0, 1).numpy(), data.x.numpy(), node_size=100)

        plt.show()


def try_subsampling():
        from torch_geometric.utils import to_undirected, add_remaining_self_loops, remove_self_loops
        from networks.pool import EdgePooling, GraclusMaxPool
        from networks.module import SeqFeatLayer, OfflineFeatLayer
        from torch_geometric.transforms import Cartesian
        from offline_to_graph.construct_graph import plot_graph_attention
        test_dataset = OfflineCharDataset(root='./data/', train=True)
        # test_dataset = IsoCharDataset(root='./data/', train=True)
        # test_dataset = AirCharDataset(root='./data/', train=True)
        print(len(test_dataset))

        alphabet = test_dataset._load_alphabet()
        test_loader = DataLoader(test_dataset, batch_size=1, num_workers=4, pin_memory=False, shuffle=True)
        pool = GraclusMaxPool()
        nn = OfflineFeatLayer(transform=Cartesian(False))
        for data in test_loader:
            # print('is_batch ', isinstance(data, Batch))
            # # print(data)
            # if data.y.item() != 3545:
            #     continue

            # print(alphabet[data.y.item()])
            # print(data.y.item())
            # print(data.pos)
            # print(data.pos.size())
            data = nn(data)
            pts = data.edge_attr.numpy()
            # print(data.x)
            if data.x is None:
                data.x = torch.zeros((data.pos.size(0),))
            # print(pts)
            # pts[:, 1] = 28 - pts[:, 1]
            # print(pts.min(), pts.max())

            data = pool(data).detach()
            data.edge_index, data.edge_attr = remove_self_loops(data.edge_index, data.edge_attr)
            plt.subplot(1, 4, 1)
            print(data)
            plot_graph_attention(data.pos.numpy(), data.edge_index.transpose(0, 1).numpy(), data.x[:,0].numpy(), node_size=100)

            plt.subplot(1, 4, 2)
            data =pool(data).detach()
            print(data)
            data.edge_index, data.edge_attr = remove_self_loops(data.edge_index, data.edge_attr)
            plot_graph_attention(data.pos.numpy(), data.edge_index.transpose(0, 1).numpy(), data.x[:,0].numpy(), node_size=100)

            plt.subplot(1, 4, 3)
            data = pool(data).detach()
            print(data)
            data.edge_index, data.edge_attr = remove_self_loops(data.edge_index, data.edge_attr)
            plot_graph_attention(data.pos.numpy(), data.edge_index.transpose(0, 1).numpy(), data.x[:,0].numpy(), node_size=100)

            plt.subplot(1, 4, 4)
            data = pool(data).detach()
            print(data)
            data.edge_index, data.edge_attr = remove_self_loops(data.edge_index, data.edge_attr)
            plot_graph_attention(data.pos.numpy(), data.edge_index.transpose(0, 1).numpy(), data.x[:,0].numpy(), node_size=100)
            # plt.subplot(122)
            # # data = transform(data)
            # pts = data.pos.numpy()
            # print(pts.min(), pts.max())
            # plot_graph(pts, data.edge_index.transpose(0, 1).numpy())
            plt.tight_layout()
            plt.show()


def check_data():
    from offline_to_graph.construct_graph import plot_graph_attention
    from networks.model import OffGCN6
    from tqdm import tqdm

    device = torch.device('cuda:0')
    dset = OfflineCharDataset(root='./data/', train=True)
    dloader = DataLoader(dset, batch_size=200, shuffle=True, num_workers=4)
    model = OffGCN6(3755).to(device)
    dloader = dset

    n_nodes = []
    for data in tqdm(dloader):
        n_nodes.append(data.x.size(0))
        # data = data.to(device)
        # print(data.x.size())
        # print(data.x.cpu().numpy())
        # pred = model(data).max(1)[1]
        # print(pred.size())



def try_3755dict():
    lb_dict = pickle.load(open('data/char/3755dict.pkl', 'rb'))
    print(len(lb_dict))
    print(min(lb_dict.keys()))
    print(max(lb_dict.keys()))
    print(min(lb_dict.values()))
    print(max(lb_dict.values()))
    print(lb_dict)


def try_ckpt_dict():
    import torch
    ckpt = './ckpt_jnl/net5-airchar-97.05/best.ckpt'
    state_dict = torch.load(ckpt, map_location='cpu')
    # print(state_dict)
    for key, val in state_dict.items():
        print(key)


def try_featlayer():
    from networks.module import SeqFeatLayer

    test_dataset = IsoCharDataset(root='./data/', train=False)
    print(len(test_dataset))
    alphabet = test_dataset._load_alphabet()
    featlayer = SeqFeatLayer(spatial=False, temporal=True)
    print(featlayer)
    for data in test_dataset:
        data = Batch.from_data_list([data])

        print(data)
        data = featlayer(data)
        pts = data.pos.numpy()
        print(pts.min(), pts.max())
        print(data.x.size())
        print(data.x.cpu().numpy())
        # plot_graph_attention(pts, data.edge_index.transpose(0, 1).numpy())
        # plt.show()
        input('next')


def try_net():
    import torch
    from networks import SEM_GCNet, UCAS_Abl_GCNet, CASIA_Abl_GCNet
    from utils import _info

    # device = torch.device('cuda')
    # test_dataset = EMnistSTkeletons(root='./data/', set_name='balanced', train=False)
    # test_loader = DataLoader(test_dataset, batch_size=10, num_workers=4)
    # net = SEM_GCNet(test_dataset.num_classes, 3).to(device)
    # for batch in test_loader:
    #     out = net(batch.to(device))
    #     print(out.shape)
    #     input('press any key to continue.')
    net = CASIA_Abl_GCNet(depth=3)
    _info(net)


def try_dynamic_knn():
    from offline_to_graph.construct_graph import plot_graph_attention
    from networks.module import BuildKnnEdge
    from torch_geometric.data.batch import Batch
    test_dataset = SuperPixelCharDataset(root='./data/', train=False)
    print(len(test_dataset))
    alphabet = test_dataset._load_alphabet()
    knn_edge_builder = BuildKnnEdge(knn=20)
    data_list = []
    data_points = []
    for data in test_dataset:
        data_list.append(data)
        data_points.append(data.pos.size(0))
        if len(data_list) == 2:
            batch = Batch.from_data_list(data_list)
            print('is_batch ', isinstance(batch, Batch))
            batch = knn_edge_builder(batch)
            print(batch)
            data = batch
            n_points_0 = data_points[0]
            data.pos[n_points_0:, 0] = 2 + data.pos[n_points_0:, 0]
            print(data)
            print(alphabet[data.y[0].item()], ' ', alphabet[data.y[1].item()])
            print(data.pos.size())
            pts = data.pos.numpy()
            print(pts.min(), pts.max())
            plot_graph_attention(pts, data.edge_index.transpose(0, 1).numpy(), data.x.squeeze(), node_size=100)
            plt.show()
            data_list = []
            data_points = []


def try_sparse():
    import torch
    import itertools
    from torch_sparse import spspmm

    device = torch.device('cuda:0')

    def block_diag_sparse(*arrs):
        bad_args = [k for k in range(len(arrs)) if not (isinstance(arrs[k], torch.Tensor) and arrs[k].ndim == 2)]
        if bad_args:
            raise ValueError("arguments in the following positions must be 2-dimension tensor: %s" % bad_args)

        shapes = torch.tensor([a.shape for a in arrs])

        i = []
        v = []
        r, c = 0, 0
        for k, (rr, cc) in enumerate(shapes):
            i += [torch.LongTensor(list(itertools.product(np.arange(c, c + cc), np.arange(r, r + rr)))).t()]
            v += [arrs[k].flatten()]
            r += rr
            c += cc
        if arrs[0].device == "cpu":
            out = torch.sparse.DoubleTensor(torch.cat(i, dim=1), torch.cat(v), torch.sum(shapes, dim=0).tolist())
        else:
            out = torch.cuda.sparse.DoubleTensor(torch.cat(i, dim=1).to(device), torch.cat(v),
                                                 torch.sum(shapes, dim=0).tolist())
        return out

    a = torch.tensor([
        [1, 2],
        [3, 4]
    ]).to(device)
    b = torch.tensor([
        [5, 6],
        [7, 8]
    ]).to(device)
    c = block_diag_sparse(a, b)
    print(c)
    print(c.to_dense())
    # c2 = spspmm(c.indices, c.values, c.indices, c.values, c.size(0), c.size(1), c.size(1))
    print(torch.sparse.mm(c, c))


def try_attention():
    from copy import deepcopy
    from torch_geometric.utils import to_dense_batch, to_undirected, add_remaining_self_loops
    from networks.module import Attention, GraphAttention, GraclusMaxPool
    device = torch.device('cuda:0')
    # test_dataset = IsoCharDataset(root='./data/', train=False)
    test_dataset = OfflineCharDataset(root='./data/', train=False)
    print(len(test_dataset))
    bz = 2
    test_loader = DataLoader(test_dataset, batch_size=bz, num_workers=1, shuffle=False)
    # attn_layer = Attention(D=bz*2).to(device)
    attn_layer = GraphAttention(in_dim=2, n_heads=2).to(device)
    max_pool = GraclusMaxPool().to(device)
    for i, data in enumerate(test_loader):
        data = data.to(device)
        data.edge_index = to_undirected(data.edge_index)
        data.edge_index, _ = add_remaining_self_loops(data.edge_index, num_nodes=data.x.size(0))
        data.x = data.pos
        print('before max_pool: ', data.x.size())
        data = max_pool(data)
        print('after max_pool: ', data.x.size())
        print('before max_pool: ', data.x.size())
        data = max_pool(data)
        print('after max_pool: ', data.x.size())
        print('before max_pool: ', data.x.size())
        data = max_pool(data)
        print('after max_pool: ', data.x.size())
        print('before max_pool: ', data.x.size())
        data = max_pool(data)
        print('after max_pool: ', data.x.size())
        print('before max_pool: ', data.x.size())
        data = max_pool(data)
        print('after max_pool: ', data.x.size())
        print('old:\r\n', data.x)
        data= attn_layer(data)
        print('new:\r\n', data.x)
        # dense_x, dense_mask = to_dense_batch(data.pos, data.batch, -1)
        # dense_x = dense_x.repeat(1, 1, bz)
        # print(dense_x)
        # print(dense_mask)
        # print(dense_x.device, dense_x.size(), type(dense_x))
        # # attn_x = attn_layer(dense_x, dense_mask)
        # attn_x = deepcopy(dense_x)
        # print(dense_x.size(), attn_x.size(), dense_mask.size())
        # new_x = attn_x.masked_select(dense_mask.unsqueeze(dim=-1)).view(data.pos.size(0), -1)
        # print(data.pos.size(), new_x.size())
        # print('|new_x - data.pos|=', (new_x[:, :data.pos.size(-1)] - data.pos).abs().sum().item())
        input('')


if __name__ == '__main__':
    # try_chars()
    # try_digits()
    # try_3755dict()
    # try_ckpt_dict()
    # try_dataloader()
    try_char_dataloader()
    # try_subsampling()
    # try_attention()
    # check_data()
    # debug_multiprocess()
    # try_plot_graph()
    # try_featlayer()
    # try_net()
    # try_info_net()
    # try_char_dataloader()
    # try_dynamic_knn()
    # try_sparse()
