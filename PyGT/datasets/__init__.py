from datasets.hcc import OfflineCharDataset, OnlineCharDataset, OfflineImageDataset,  GenOfflineCharDataset
from datasets.mnist import GridMNIST, MNISTSuperpixels, MnistSkeletons


AllDatasets = {'on_hccr': OnlineCharDataset,
               'off_hccr': OfflineCharDataset,
               'gen_off_hccr': GenOfflineCharDataset,
               'mnist' : MnistSkeletons}
