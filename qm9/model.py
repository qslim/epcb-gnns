import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, ELU, Sigmoid
from utils.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn import global_add_pool, global_mean_pool
from conv import GinConv, ExpC, CombC, ExpC_star, CombC_star


class Net(torch.nn.Module):
    def __init__(self,
                 dataset,
                 config):
        super(Net, self).__init__()
        self.pooling = config.pooling
        self.lin0 = Linear(dataset.num_features, config.hidden)

        self.convs = torch.nn.ModuleList()
        if config.methods[:2] == 'EB':
            for i in range(config.layers):
                self.convs.append(ExpC(config.hidden,
                                                 int(config.methods[2:]),
                                                 config.variants))
        elif config.methods[:2] == 'EA':
            for i in range(config.layers):
                self.convs.append(ExpC_star(config.hidden,
                                                 int(config.methods[2:]),
                                                 config.variants))
        elif config.methods == 'CB':
            for i in range(config.layers):
                self.convs.append(CombC(config.hidden, config.variants))
        elif config.methods == 'CA':
            for i in range(config.layers):
                self.convs.append(CombC_star(config.hidden, config.variants))
        elif config.methods == 'GIN':
            for i in range(config.layers):
                self.convs.append(GinConv(config.hidden, config.variants))
        else:
            raise ValueError('Undefined gnn layer called {}'.format(config.methods))

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
