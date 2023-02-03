import os, glob, h5py
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Dataset
from tqdm import tqdm
import random


class OnlineCharDataset(InMemoryDataset):
    sub_dir = '/on-hccr/'

    def __init__(self, root, train:bool, transform=None):
        super(OnlineCharDataset, self).__init__(root + self.sub_dir, transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['/home/ganji/TmpData/CASIA-POT/train/graphs_s/',
                '/home/ganji/TmpData/CASIA-POT/test/graphs_s/',
                '3755.txt']

    @property
    def processed_file_names(self):
        return ['train_on_hccr.pt', 'test_on_hccr.pt']

    def _load_alphabet(self):
        alphabet = []
        with open(self.raw_paths[2]) as f:
            for line in f.readlines():
                label = line.split('\t')[0]
                alphabet.append(label)
        return alphabet

    def download(self):
        pass

    def process(self):
        for i, gpath in enumerate(self.raw_file_names):
            data_list = []
            files = list(glob.glob(gpath + '*.dat'))
            for fn in tqdm(files, desc='%s'%gpath):
                datas = torch.load(fn)
                data_list.extend(datas)

            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[i])
            print('save to ', self.processed_paths[i])


class OfflineImageDataset(Dataset):
    IMG_H5_ROOT = '/home/ganji/TmpData/CASIA-GNT-H5/h5_64/'
    num_classes = 3755

    def __init__(self, split='train', shuffled=False):
        self.data_path = os.path.join(self.IMG_H5_ROOT, split)
        imgs, lbs = [], []
        for f in glob.glob(self.data_path + '/*.h5'):
            img, lb = self._load_h5(f)
            imgs.append(img)
            lbs.append(lb)

        self.imgs = np.concatenate(imgs, axis=0)
        self.lbs = np.concatenate(lbs, axis=0)

        if shuffled:
            idxs = list(range(self.imgs.shape[0]))
            random.shuffle(idxs)
            self.imgs = self.imgs[idxs]
            self.lbs = self.lbs[idxs]

    def _load_h5(self, f):
        h5_f = h5py.File(f, 'r')
        fmaps = h5_f['imgs'][:].astype('uint8')
        labels = h5_f['lbs'][:]
        h5_f.close()
        return fmaps, labels

    def __len__(self):
        return self.lbs.shape[0]

    def __getitem__(self, item):
        return self.imgs[item], self.lbs[item]

    def _load_alphabet(self, path):
        alphabet = []
        with open(path) as f:
            for line in f.readlines():
                label = line.split('\t')[0]
                alphabet.append(label)
        return alphabet


class OfflineSkeletonImageDataset(Dataset):
    IMG_SIZE = 72
    SKE_H5_ROOT = '/home/ganji/TmpData/CASIA-CSKE-H5/h5_%d/'%IMG_SIZE

    def __init__(self, split='train'):
        self.data_path = os.path.join(self.SKE_H5_ROOT, split)
        imgs, skes, lbs = [], [], []
        for fn in glob.glob(self.data_path + '/*.h5'):
            img, ske, lb = self.load_h5(fn)
            imgs.append(img)
            skes.append(ske)
            lbs.append(lb)

        self.imgs = np.concatenate(imgs, axis=0)
        self.skes = np.concatenate(skes, axis=0)
        self.lbs = np.concatenate(lbs, axis=0)

    def load_h5(self, fn):
        h5_f = h5py.File(fn, 'r')
        imgs = h5_f['imgs'][:].astype(np.uint8)
        fmaps = h5_f['fmaps'][:]
        labels = h5_f['lbs'][:]
        h5_f.close()
        return imgs, fmaps, labels

    def __len__(self):
        return self.lbs.shape[0]

    def __getitem__(self, item):
        return self.imgs[item], self.skes[item], self.lbs[item]


class OfflineCharDataset(InMemoryDataset):
    sub_dir = '/off-hccr/'
    img_size = 64

    def __init__(self, root, train:bool, transform=None):
        super(OfflineCharDataset, self).__init__(root + self.sub_dir, transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
            return ['/home/ganji/TmpData/CASIA-GRAPH-H5/h5_%d/train'%self.img_size,
                    '/home/ganji/TmpData/CASIA-GRAPH-H5/h5_%d/test'%self.img_size,
                    '3755.txt']

    @property
    def processed_file_names(self):
        return ['training_off_hccr_%d.pt'%self.img_size, 'test_off_hccr_%d.pt'%self.img_size]

    def _load_alphabet(self):
        alphabet = []
        with open(self.raw_paths[2]) as f:
            for line in f.readlines():
                label = line.split('\t')[0]
                alphabet.append(label)
        return alphabet

    def download(self):
        pass

    def process(self):
        for i, gpath in enumerate(self.raw_file_names):
            data_list = []
            files = list(glob.glob(gpath + '/*.h5'))
            for fn in tqdm(files, desc='%s'%gpath):
                samples = torch.load(fn)
                for sample in samples:
                    sample.x = sample.x.to(torch.float16)
                    sample.pos = sample.pos.to(torch.float16)
                    sample.edge_index = sample.edge_index.to(torch.int16)
                    data_list.append(sample)

            random.shuffle(data_list)
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[i])
            print('save to ', self.processed_paths[i])

    def __getitem__(self, item):
        data = super(OfflineCharDataset, self).__getitem__(item)
        # data.pos = 2 * data.pos / self.img_size - 1
        data.x = data.x.to(torch.float32)
        data.pos = data.pos.to(torch.float32)
        data.edge_index = data.edge_index.to(torch.long)
        return data


class GenOfflineCharDataset(InMemoryDataset):
    sub_dir = '/gen-off-hccr/'
    img_size = 64

    def __init__(self, root, train:bool, transform=None):
        super(GenOfflineCharDataset, self).__init__(root + self.sub_dir, transform)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
            return ['/home/ganji/TmpData/CASIA-GRAPH-H5-GEN/h5_%d/train'%self.img_size,
                    '/home/ganji/TmpData/CASIA-GRAPH-H5-GEN/h5_%d/test'%self.img_size,
                    '3755.txt']

    @property
    def processed_file_names(self):
        return ['training_gen_off_hccr_%d.pt'%self.img_size, 'test_gen_off_hccr_%d.pt'%self.img_size]

    def _load_alphabet(self):
        alphabet = []
        with open(self.raw_paths[2]) as f:
            for line in f.readlines():
                label = line.split('\t')[0]
                alphabet.append(label)
        return alphabet

    def download(self):
        pass

    def process(self):
        for i, gpath in enumerate(self.raw_file_names):
            data_list = []
            files = list(glob.glob(gpath + '/*.h5'))
            for fn in tqdm(files, desc='%s'%gpath):
                samples = torch.load(fn)
                for sample in samples:
                    sample.x = sample.x.to(torch.float16)
                    sample.pos = sample.pos.to(torch.float16)
                    sample.edge_index = sample.edge_index.to(torch.int16)
                    data_list.append(sample)

            random.shuffle(data_list)
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[i])
            print('save to ', self.processed_paths[i])

    def __getitem__(self, item):
        data = super(GenOfflineCharDataset, self).__getitem__(item)
        # data.pos = 2 * data.pos / self.img_size - 1
        data.x = data.x.to(torch.float32)
        data.pos = data.pos.to(torch.float32)
        data.edge_index = data.edge_index.to(torch.long)
        return data
