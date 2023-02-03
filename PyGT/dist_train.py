import argparse, shutil
from datetime import datetime
import os
import torch.distributed as dist
from torch.optim.lr_scheduler import MultiStepLR
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from networks.model import *
from networks.utils import _info
from datasets import AllDatasets as Datasets
from networks import AllNetworks as Networks


def get_config():
    parser = argparse.ArgumentParser(description='PyTorch graph classification')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='training epochs')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='initial learning rate')
    parser.add_argument('--grad-clip', type=float, default=10.,
                        help='gradient clip')
    parser.add_argument('--root', type=str, default='./data/',
                        help='dataset directory (default "./data/")')
    parser.add_argument('--dataset', type=str, default='char',
                        help='dataset name ("digit" or "char")')
    parser.add_argument('--net-id', type=str, default='1')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--load-ckpt', type=str)
    parser.add_argument('--save-ckpt', type=str, default='')
    parser.add_argument('--pop-ckpt-key', type=str, default='')
    parser.add_argument('--set-name', type=str, default='')
    parser.add_argument('--first-decay', type=float, default=0.7)
    parser.add_argument('--last-decay', type=float, default=0.9)
    parser.add_argument('--ckpt-root', type=str, default='./ckpt/')
    parser.add_argument('--in-features', type=int, default=0)
    parser.add_argument('--online', type=int, default=1)
    parser.add_argument('--save-iter', type=int, default=0)

    ## multi-gpu training settings
    parser.add_argument('--init-method', type=str, default='tcp://127.0.0.1:12345')
    parser.add_argument('--backbend', type=str, default='nccl')
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--world-size', type=int, default=2)

    args = parser.parse_args()
    return args


def test(model, dataloader, device, print_detail=False):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for i, data in enumerate(dataloader):
            data = data.to(device)
            pred = model(data).max(1)[1]
            correct += pred.eq(data.y).sum().item()
            total += len(data.y)
            del data
    test_acc = correct / len(dataloader.dataset) * 100.
    model.train()
    torch.cuda.empty_cache()
    return test_acc, (correct, total)


def main():
    args = get_config()
    local_rank = int(os.environ["LOCAL_RANK"])
    dist.init_process_group(backend=args.backbend)
    print('[dist init] %s'%torch.distributed.is_initialized())
    gpu_id = local_rank
    Dataset = Datasets[args.dataset]
    transform = None

    train_dataset = Dataset(root=args.root, train=True, transform=transform)
    test_dataset = Dataset(root=args.root, train=False, transform=transform)

    print(Dataset, train_dataset.num_classes) if local_rank == 0 else ''

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.workers, pin_memory=False)

    device = torch.device('cuda:%d'%gpu_id)
    if args.in_features == 0:
        args.in_features = train_dataset.num_node_features
    model = Networks[args.net_id](num_classes=train_dataset.num_classes,
                                  in_features=args.in_features,
                                  online=args.online).to(device).float()
    if args.load_ckpt:
        state_dict = torch.load(args.load_ckpt, map_location=device)
        if not args.pop_ckpt_key:
            model.load_state_dict(state_dict)
        else:
            pop_keys = args.pop_ckpt_key.strip().split(',')
            for key in pop_keys:
                state_dict.pop(key)
            print('Pop state_dict keys: ', pop_keys)
            model.load_state_dict(state_dict, strict=False)
        if gpu_id == 0:
            print('Load pretrained: ', args.load_ckpt)
            test_acc, (correct, total) = test(model, test_loader, device, print_detail=True)
            print('Test Accuracy: {:.2f}%%  ({} / {})'.format(test_acc, correct, total))

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_id], output_device=gpu_id)

    if args.save_iter == 0:
        args.save_iter = len(train_loader)

    if gpu_id == 0:
        _info(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = MultiStepLR(optimizer,
                               milestones=[int(args.epochs * args.first_decay) - 1,
                                           int(args.epochs * args.last_decay) - 1],
                               gamma=0.1)

    if len(args.save_ckpt.strip()) > 0:
        time_stamp = args.save_ckpt.strip()
    else:
        time_stamp = args.ckpt_root + datetime.strftime(datetime.now(), '%m-%d-%H-%M-%S') + \
                     '-net%s-%s'%(args.net_id, args.dataset) + args.set_name

    best_acc = 0.
    writer = None
    num_batchs = len(train_loader)
    total_iters = 0
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()

        with tqdm(enumerate(train_loader), total=len(train_loader)) as tbar:
            for i, data in tbar:
                total_iters += 1
                data = data.to(device, non_blocking=False)
                optimizer.zero_grad()
                loss = F.nll_loss(model(data), data.y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                if writer and gpu_id == 0:
                    writer.add_scalar('loss', loss.item(), i + epoch * num_batchs)

                if gpu_id == 0:
                    info = 'Epoch: %2d Iter: %d Loss: %f' %(epoch, i, loss.item())
                    tbar.set_description(info)

                if total_iters % args.save_iter == 0:
                    test_acc, (correct, total) = test(model, test_loader, device, print_detail=False)
                    print('Rank:{} Test Accuracy: {:.2f}%%  ({} / {})'.format(gpu_id, test_acc, correct, total))
                    model.train()
                    if writer is None:
                        writer = SummaryWriter()

                    writer.add_scalar('test_acc_%d'%gpu_id, test_acc, total_iters // args.save_iter)

                    if best_acc < test_acc:
                        best_acc = test_acc
                        if not os.path.exists(time_stamp):
                            os.makedirs(time_stamp)
                            with open(time_stamp + '/model.txt', 'w') as f:
                                f.writelines(_info(model.module, ret_str=True))
                            shutil.copy('./networks/model.py', time_stamp)

                        torch.save(model.module.state_dict(), '%s/best_%d.ckpt'%(time_stamp, gpu_id))
                        with open(time_stamp + '/best_acc_%d.txt'%gpu_id, 'w') as f:
                            f.writelines('GCNNET%s \nbatch-size %d\nbest_acc %.2f%%'%(args.net_id, args.batch_size, best_acc))

                del data, loss

        lr_scheduler.step(epoch)

    test_acc, (correct, total) = test(model, test_loader, device, print_detail=False)
    print('Rank:{} Test Accuracy: {:.2f}%%  ({} / {})'.format(gpu_id, test_acc, correct, total))
    if best_acc < test_acc and os.path.exists(time_stamp):
        torch.save(model.module.state_dict(), '%s/best_%d.ckpt'%(time_stamp, gpu_id))
        with open(time_stamp + '/best_acc_%d.txt'%gpu_id, 'w') as f:
            f.writelines('GCNNET%s \nbatch-size %d\nbest_acc %.2f%%' % (args.net_id, args.batch_size, best_acc))


if __name__ == '__main__':
    main()
