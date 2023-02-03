import os
import numpy as np
from torch_geometric.data import InMemoryDataset, Data, extract_zip
from torch_geometric.utils import grid
from tqdm import tqdm
from preprocess.offline_to_graph.construct_graph import image2graph
import torch
from torchvision.datasets.mnist import MNIST
try:
    from torchvision.datasets.utils import download_url, download_and_extract_archive, extract_archive, makedir_exist_ok, verify_str_arg
except ImportError:
    pass



class GridMNIST(InMemoryDataset):
    training_file, test_file = 'training.pt', 'test.pt'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
               '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']
    grid_size = 28
    sub_dir = '/mnist_grid/'

    def __init__(self, root, train=True, transform=None, pre_transform=None):
        super(GridMNIST, self).__init__(root + self.sub_dir, transform, pre_transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [self.training_file, self.test_file]

    @property
    def processed_file_names(self):
        return [self.training_file, self.test_file]

    @property
    def num_classes(self):
        return 10

    def get_raw_dataset(self):
        train_dataset = MNIST(root=self.root + '/../', train=True, download=True)
        test_dataset = MNIST(root=self.root + '/../', train=False, download=True)
        return train_dataset, test_dataset

    def _load_alphabet(self):
        dict = {}
        for i in range(10):
            dict[i] = self.classes[i]
        return dict

    def download(self):
        self.get_raw_dataset()

    def process(self):
        train_dataset, test_dataset = self.get_raw_dataset()
        edge_index, pos = grid(self.grid_size, self.grid_size)

        def _process(dset, spath):
            data_list = []
            for img, target in tqdm(dset, desc='%s'%spath):
                np_img = np.array(img).flatten()
                data = Data(x=torch.from_numpy(np_img).float() / 128. - 1.0,
                            edge_index=edge_index,
                            pos=pos,
                            y=torch.LongTensor([target]))
                data_list.append(data)
            data, slices = self.collate(data_list)
            torch.save((data, slices), spath)
            print('save to ', spath)

        _process(test_dataset, self.processed_paths[1])
        _process(train_dataset, self.processed_paths[0])


class MNISTSuperpixels(InMemoryDataset):
    r"""MNIST superpixels dataset from the `"Geometric Deep Learning on
    Graphs and Manifolds Using Mixture Model CNNs"
    <https://arxiv.org/abs/1611.08402>`_ paper, containing 70,000 graphs with
    75 nodes each.
    Every graph is labeled by one of 10 classes.

    Args:
        root (string): Root directory where the dataset should be saved.
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://data.pyg.org/datasets/MNISTSuperpixels.zip'
    sub_dir = '/mnist_spix/'
    classes = ['0 - zero', '1 - one', '2 - two', '3 - three', '4 - four',
                   '5 - five', '6 - six', '7 - seven', '8 - eight', '9 - nine']

    def __init__(self, root, train=True, transform=None, pre_transform=None,
                 pre_filter=None):
        super(MNISTSuperpixels, self).__init__(root + self.sub_dir, transform, pre_transform,
                                               pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['MNISTSuperpixels.pt']

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)

    def _load_alphabet(self):
        dict = {}
        for i in range(10):
            dict[i] = self.classes[i]
        return dict

    def process(self):
        inputs = torch.load(self.raw_paths[0])
        for i in range(len(inputs)):
            data_list = [Data(**data_dict) for data_dict in inputs[i]]

            if self.pre_filter is not None:
                data_list = [d for d in data_list if self.pre_filter(d)]

            if self.pre_transform is not None:
                data_list = [self.pre_transform(d) for d in data_list]

            torch.save(self.collate(data_list), self.processed_paths[i])


class MnistSkeletons(InMemoryDataset):
    img_size = 56
    merge_dist_thres = 1.7
    half_win_size = 1
    sub_dir = '/mnist_skeleton/'
    edge_order = 1
    training_file, test_file = 'training.%dorder.pt'%edge_order, 'test.%dorder.pt'%edge_order
    mean_val_file = 'mean_node_val.txt'
    pre_processed = False
    debug = False

    def __init__(self, root, train=True, transform=None, pre_transform=None):
        super(MnistSkeletons, self).__init__(root + self.sub_dir, transform, pre_transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)
        # self.mean = float(np.loadtxt(self.processed_paths[2]))

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return [self.training_file, self.test_file, self.mean_val_file]

    @property
    def num_classes(self):
        return 10

    def _load_alphabet(self):
        dict = {}
        for i in range(10):
            dict[i] = i
        return dict

    def get_raw_dataset(self, download=True):
        train_dataset = MNIST(root=self.root + '/../', train=True, download=download)
        test_dataset = MNIST(root=self.root + '/../', train=False, download=download)
        return train_dataset, test_dataset

    def download(self):
        self.get_raw_dataset(True)

    def process(self):
        train_dataset, test_dataset = self.get_raw_dataset(False)

        def _my_process(dset, spath, avg_val_path=None):
            if os.path.exists(spath):
                return
            data_list = []
            avg_val = np.zeros((3,)).astype(np.float32)
            avg_count = 0
            for img, target in tqdm(dset, desc='%s'%spath):
                nodes, edges, node_vals = image2graph(img, width=self.img_size, merge_dist_thres=self.merge_dist_thres,
                                                      half_win_size=self.half_win_size, edge_order=self.edge_order,
                                                      with_processing=self.pre_processed, debug=self.debug)
                data = Data(x=torch.Tensor(node_vals).float().unsqueeze(dim=-1),
                            edge_index=torch.LongTensor(edges).transpose(0, 1),
                            pos=torch.Tensor(nodes).float(),
                            y=torch.LongTensor([target]))
                data_list.append(data)

                avg_val[2] += node_vals.sum()
                avg_val[:2] += nodes.sum(axis=0)
                avg_count += node_vals.shape[0]

            data, slices = self.collate(data_list)
            torch.save((data, slices), spath)
            print('save to ', spath)

            avg_val /= avg_count
            if avg_val_path:
                np.savetxt(avg_val_path, avg_val, fmt='%.6f')

        _my_process(test_dataset, self.processed_paths[1])
        _my_process(train_dataset, self.processed_paths[0], self.processed_paths[2])

