import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, ELU, Sigmoid
from utils.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn import global_add_pool, global_mean_pool
from conv import GinConv, ExpandingBConv, CombBConv, ExpandingAConv, CombAConv


class Net(torch.nn.Module):
    def __init__(self,
                 dataset,
                 config):
        super(Net, self).__init__()
        self.pooling = config.pooling
        self.lin0 = Linear(dataset.num_features, config.hidden)

        self.convs = torch.nn.ModuleList()
        if config.nonlinear_conv[:2] == 'EB':
            for i in range(config.layers):
                self.convs.append(ExpandingBConv(config.hidden,
                                                 int(config.nonlinear_conv[2:]),
                                                 config.variants))
        elif config.nonlinear_conv[:2] == 'EA':
            for i in range(config.layers):
                self.convs.append(ExpandingAConv(config.hidden,
                                                 int(config.nonlinear_conv[2:]),
                                                 config.variants))
        elif config.nonlinear_conv == 'CB':
            for i in range(config.layers):
                self.convs.append(CombBConv(config.hidden, config.variants))
        elif config.nonlinear_conv == 'CA':
            for i in range(config.layers):
                self.convs.append(CombAConv(config.hidden, config.variants))
        elif config.nonlinear_conv == 'GIN':
            for i in range(config.layers):
                self.convs.append(GinConv(config.hidden, config.variants))
        else:
            ValueError('Undefined conv called {}'.format(config.nonlinear_conv))

        self.JK = JumpingKnowledge(config.JK)
        if config.JK == 'cat':
            self.lin1 = torch.nn.Linear(config.layers * config.hidden, (config.layers + 1) // 2 * config.hidden)
            self.lin2 = torch.nn.Linear((config.layers + 1) // 2 * config.hidden, config.hidden)
        else:
            self.lin1 = torch.nn.Linear(config.hidden, config.hidden)
            self.lin2 = torch.nn.Linear(config.hidden, config.hidden)
        self.lin3 = torch.nn.Linear(config.hidden, 1)

        if config.pooling == 'add':
            self.pool = global_add_pool
        elif config.pooling == 'mean':
            self.pool = global_mean_pool

    def reset_parameters(self):
        self.lin0.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = self.lin0(x)
        xs = []

        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            xs += [self.pool(x, batch)]

        x = self.JK(xs)
        x = F.elu(self.lin1(x))
        x = F.elu(self.lin2(x))
        x = self.lin3(x)
        return x.view(-1)

    def __repr__(self):
        return self.__class__.__name__

