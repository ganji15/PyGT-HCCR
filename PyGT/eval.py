import os, argparse, shutil
from datetime import datetime
import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.data import DataLoader
from torch_geometric.transforms import *
from torch.utils.tensorboard import SummaryWriter

from networks.model import *
from networks.utils import *
from networks.utils import _info
from datasets import AllDatasets as Datasets
from networks import AllNetworks as Networks


def get_config():
    parser = argparse.ArgumentParser(description='PyTorch graph classification')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--root', type=str, default='./data/',
                        help='dataset directory (default "./data/")')
    parser.add_argument('--dataset', type=str, default='gmnist',
                        help='dataset name ("digit" or "char")')
    parser.add_argument('--set-name', type=str, default='')
    parser.add_argument('--online', type=int, default=0)
    parser.add_argument('--net-id', type=str, default='4')
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--in-features', type=int, default=0)

    args = parser.parse_args()
    return args


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed(seed)


def test(model, dataloader, device, print_detail=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data = data.to(device)
            # print(data.x.size())
            # print(data.x.cpu().numpy())
            pred = model(data).max(1)[1]
            correct += pred.eq(data.y).sum().item()
            total += len(data.y)
    test_acc = correct * 100. / len(dataloader.dataset)
    return test_acc, (correct, total)


def main():
    args = get_config()
    Dataset = Datasets[args.dataset.strip()]

    transform = None
    if args.set_name.strip():
        test_dataset = Dataset(root=args.root.strip(), train=False, transform=transform, set_name=args.set_name.strip())
    else:
        test_dataset = Dataset(root=args.root.strip(), train=False, transform=transform)

    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.in_features == 0:
        args.in_features = test_dataset.num_node_features
    model = Networks[args.net_id.strip()](num_classes=test_dataset.num_classes,
                                  in_features=args.in_features, online=args.online).to(device)

    # set_random_seed(12345)
    _info(model, detail=False)
    if args.ckpt:
        model.load_state_dict(torch.load(args.ckpt.strip()))
    test_acc, (correct, total) = test(model, test_loader, device, print_detail=True)
    print('Test Accuracy: {:.2f}%  ({} / {})'.format(test_acc, correct, total))


if __name__ == '__main__':
    main()
