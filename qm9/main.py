import os.path as osp
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter

import sys
sys.path.append('..')

from utils.qm9 import QM9
from model import Net
from utils.config import process_config, get_args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyTransform(object):
    def __call__(self, data):
        data.y = data.y[:, int(config.target)]
        return data


def train():
    model.train()
    loss_all = 0

    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        loss = F.mse_loss(model(data), data.y)
        loss.backward()
        loss_all += loss * data.num_graphs
        optimizer.step()
    return loss_all / len(train_loader.dataset)


def test(loader):
    model.eval()
    error = 0

    for data in loader:
        data = data.to(device)
        error += ((model(data) * std[config.target].cuda()) -
                  (data.y * std[config.target].cuda())).abs().sum().item()
    return error / len(loader.dataset)


args = get_args()
config = process_config(args)
print(config)

if config.get('seed') is not None:
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)


path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', 'QM9')
dataset = QM9(path, transform=T.Compose([MyTransform(), T.Distance()]))
dataset = dataset.shuffle()

# Normalize targets to mean = 0 and std = 1.
tenpercent = int(len(dataset) * 0.1)
mean = dataset.data.y[tenpercent * 2:].mean(dim=0)
std = dataset.data.y[tenpercent * 2:].std(dim=0)
dataset.data.y = (dataset.data.y - mean) / std

test_dataset = dataset[:tenpercent]
val_dataset = dataset[tenpercent:tenpercent * 2]
train_dataset = dataset[tenpercent * 2:]
test_loader = DataLoader(test_dataset, batch_size=config.hyperparams.batch_size)
val_loader = DataLoader(val_dataset, batch_size=config.hyperparams.batch_size)
train_loader = DataLoader(train_dataset, batch_size=config.hyperparams.batch_size, shuffle=True)

model = Net(dataset, config.architecture).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=config.hyperparams.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                            step_size=config.hyperparams.step_size,
                                            gamma=config.hyperparams.decay_rate)

ts_algo_hp = str(config.time_stamp) + '_' \
             + str(config.commit_id[0:7]) + '_' \
             + str(config.architecture.methods) + '_' \
             + str(config.architecture.variants.fea_activation) + '_' \
             + str(config.architecture.pooling) + '_' \
             + str(config.architecture.JK) + '_' \
             + str(config.architecture.layers) + '_' \
             + str(config.architecture.hidden) + '_' \
             + str(config.architecture.variants.BN) + '_' \
             + str(config.hyperparams.learning_rate) + '_' \
             + str(config.hyperparams.step_size) + '_' \
             + str(config.hyperparams.decay_rate) + '_' \
             + 'B' + str(config.hyperparams.batch_size) + '_' \
             + 'S' + str(config.seed)

print('--------')
print('QM9_' + str(config.target) + ', '
      + ts_algo_hp
      + ', ID=' + config.commit_id)

writer = SummaryWriter(config.directory)

best_val_error = None
for epoch in range(1, config.hyperparams.epochs):
    lr = scheduler.optimizer.param_groups[0]['lr']
    loss = train()
    val_error = test(val_loader)
    scheduler.step()

    if best_val_error is None:
        best_val_error = val_error
    test_error = test(test_loader)
    if val_error <= best_val_error:
        best_val_error = val_error
        print(
            'Epoch: {:03d}, LR: {:7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
            'Test MAE: {:.7f}'.format(epoch, lr, loss, val_error, test_error))
    else:
        print(
            'Epoch: {:03d}, {:7f},{:.7f},{:.7f},'
            '{:.7f}'.format(epoch, lr, loss, val_error, test_error))

    writer.add_scalars(config.dataset_name + '_' + str(config.target), {ts_algo_hp + '/lr': lr}, epoch)
    writer.add_scalars(config.dataset_name + '_' + str(config.target), {ts_algo_hp + '/te': test_error}, epoch)
    writer.add_scalars(config.dataset_name + '_' + str(config.target), {ts_algo_hp + '/ve': val_error}, epoch)
    writer.add_scalars(config.dataset_name + '_' + str(config.target), {ts_algo_hp + '/ls': loss}, epoch)

writer.close()
