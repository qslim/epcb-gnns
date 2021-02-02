import torch
import random
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter

import sys
sys.path.append('..')

from model import Net
from sklearn.model_selection import StratifiedKFold
import numpy as np
import os.path as osp
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import degree
import torch_geometric.transforms as T

from utils.config import process_config, get_args


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def get_dataset(name):
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', name)
    dataset = TUDataset(path, name)
    dataset.data.edge_attr = None

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    return dataset


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train(loader, model, optimizer):
    model.train()
    loss_all = 0

    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, data.y)
        loss.backward()
        loss_all += loss.item() * data.num_graphs
        optimizer.step()
    return loss_all / len(loader.dataset)


def test(loader, model):
    model.eval()
    correct = 0

    for data in loader:
        data = data.to(device)
        pred = model(data).max(dim=1)[1]
        correct += pred.eq(data.y).sum().item()
    return correct / len(loader.dataset)


def run_given_fold(net,
                   dataset,
                   train_loader,
                   val_loader,
                   writer,
                   ts_kf_algo_hp,
                   config):
    model = net(dataset, config=config.architecture)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.hyperparams.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config.hyperparams.step_size,
                                                gamma=config.hyperparams.decay_rate)

    train_losses = []
    train_accs = []
    test_accs = []
    for epoch in range(1, config.hyperparams.epochs):
        train_loss = train(train_loader, model, optimizer)
        train_acc = test(train_loader, model)
        test_acc = test(val_loader, model)

        scheduler.step()

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print('Epoch: {:03d}, Train Loss: {:.7f}, '
              'Train Acc: {:.7f}, Test Acc: {:.7f}'.format(epoch, train_loss,
                                                           train_acc, test_acc))

        writer.add_scalars(config.dataset_name, {ts_kf_algo_hp + '/test_acc': test_acc}, epoch)
        writer.add_scalars(config.dataset_name, {ts_kf_algo_hp + '/train_acc': train_acc}, epoch)
        writer.add_scalars(config.dataset_name, {ts_kf_algo_hp + '/train_loss': train_loss}, epoch)

    return test_accs, train_losses, train_accs


def k_fold(dataset, folds, seed):
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)

    val_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        val_indices.append(torch.from_numpy(idx))

    for i in range(folds):
        train_mask = torch.ones(len(dataset), dtype=torch.uint8)
        train_mask[val_indices[i]] = 0
        train_indices.append(train_mask.nonzero().view(-1))

    return train_indices, val_indices


def run_model(net, dataset, config):
    folds_test_accs = []
    folds_train_losses = []
    folds_train_accs = []

    def k_folds_average(avg_folds):
        avg_folds = np.vstack(avg_folds)
        return np.mean(avg_folds, axis=0), np.std(avg_folds, axis=0)

    writer = SummaryWriter(config.directory)

    algo_hp = str(config.commit_id[0:7]) + '_' \
              + str(config.architecture.methods) + '_' \
              + str(config.architecture.pooling) + '_' \
              + str(config.architecture.JK) + '_' \
              + str(config.architecture.layers) + '_' \
              + str(config.architecture.hidden) + '_' \
              + str(config.architecture.variants.BN) + '_' \
              + str(config.architecture.dropout) + '_' \
              + str(config.hyperparams.learning_rate) + '_' \
              + str(config.hyperparams.step_size) + '_' \
              + str(config.hyperparams.decay_rate) + '_' \
              + 'B' + str(config.hyperparams.batch_size) + '_' \
              + 'S' + str(config.seed if config.get('seed') is not None else "na") + '_' \
              + 'W' + str(config.num_workers if config.get('num_workers') is not None else "na")

    for fold, (train_idx, val_idx) in enumerate(
            zip(*k_fold(dataset, 10, config.get('seed', 12345)))):

        if fold >= config.get('folds_cut', 10):
            break

        train_dataset = dataset[train_idx]
        val_dataset = dataset[val_idx]

        train_loader = DataLoader(train_dataset, config.hyperparams.batch_size, shuffle=True, num_workers=config.num_workers)
        val_loader = DataLoader(val_dataset, config.hyperparams.batch_size, shuffle=False, num_workers=config.num_workers)

        print('-------- FOLD' + str(fold) +
              ' DATASET=' + config.dataset_name +
              ', COMMIT_ID=' + config.commit_id)

        test_accs, train_losses, train_accs = run_given_fold(
            net,
            dataset,
            train_loader,
            val_loader,
            writer=writer,
            ts_kf_algo_hp=str(config.time_stamp) + '/f' + str(fold) + '/' + algo_hp,
            config=config
        )

        folds_test_accs.append(np.array(test_accs))
        folds_train_losses.append(np.array(train_losses))
        folds_train_accs.append(np.array(train_accs))

        # following the protocol of other GNN baselines
        avg_test_accs, std_test_accs = k_folds_average(folds_test_accs)
        sel_epoch = np.argmax(avg_test_accs)
        sel_test_acc = np.max(avg_test_accs)
        sel_test_acc_std = std_test_accs[sel_epoch]
        sel_test_with_std = str(sel_test_acc) + '_' + str(sel_test_acc_std)

        avg_train_losses, std_train_losses = k_folds_average(folds_train_losses)
        sel_tl_with_std = str(np.min(avg_train_losses)) + '_' + str(std_train_losses[np.argmin(avg_train_losses)])

        avg_train_accs, std_train_accs = k_folds_average(folds_train_accs)
        sel_ta_with_std = str(np.max(avg_train_accs)) + '_' + str(std_train_accs[np.argmax(avg_train_accs)])

        print('--------')
        print('Best Test Acc:   ' + sel_test_with_std + ', Epoch: ' + str(sel_epoch))
        print('Best Train Loss: ' + sel_tl_with_std)
        print('Best Train Acc:  ' + sel_ta_with_std)

        print('FOLD' + str(fold + 1) + ', '
              + config.dataset_name + ', '
              + str(config.time_stamp) + '/'
              + str(config.get('seed', 'NoSeed')) + '/'
              + str(config.architecture.layers) + '_'
              + str(config.architecture.hidden) + '_'
              + str(config.hyperparams.learning_rate) + '_'
              + str(config.hyperparams.step_size) + '_'
              + str(config.hyperparams.decay_rate)
              + '_B' + str(config.hyperparams.batch_size)
              + ', BT=' + sel_test_with_std
              + ', BE=' + str(sel_epoch)
              + ', ID=' + config.commit_id)

        ts_fk_algo_hp = str(config.time_stamp) + '/fk' + str(config.get('folds_cut', 10)) + '/' + algo_hp

        writer.add_scalars(config.dataset_name, {ts_fk_algo_hp + '/best_acc': sel_test_acc}, fold)
        writer.add_scalars(config.dataset_name, {ts_fk_algo_hp + '/best_std': sel_test_acc_std}, fold)
        writer.add_scalars(config.dataset_name, {ts_fk_algo_hp + '/best_epoch': sel_epoch}, fold)

    for i in range(1, config.hyperparams.epochs):
        test_acc = avg_test_accs[i - 1]
        train_loss = avg_train_losses[i - 1]
        train_acc = avg_train_accs[i - 1]

        writer.add_scalars(config.dataset_name, {ts_fk_algo_hp + '/test_acc': test_acc}, i)
        writer.add_scalars(config.dataset_name, {ts_fk_algo_hp + '/train_loss': train_loss}, i)
        writer.add_scalars(config.dataset_name, {ts_fk_algo_hp + '/train_acc': train_acc}, i)

    # writer.export_scalars_to_json("./all_scalars.json")
    writer.close()


def main():
    args = get_args()
    config = process_config(args)
    print(config)

    if config.get('seed') is not None:
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    dataset = get_dataset(config.dataset_name).shuffle()
    run_model(Net, dataset, config=config)


if __name__ == "__main__":
    main()
