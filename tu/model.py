import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, ELU, Sigmoid
from utils.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn import global_add_pool, global_mean_pool
from conv import ExpC, CombC, ExpC_star, CombC_star


class Net(torch.nn.Module):
    def __init__(self,
                 dataset,
                 config):
        super(Net, self).__init__()
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
        else:
            raise ValueError('Undefined gnn layer called {}'.format(config.methods))

        self.JK = JumpingKnowledge(config.JK)

        if config.JK == 'cat':
            self.lin1 = Linear(config.layers * config.hidden, config.hidden)
        else:
            self.lin1 = Linear(config.hidden, config.hidden)

        self.lin2 = Linear(config.hidden, dataset.num_classes)

        if config.pooling == 'add':
            self.pool = global_add_pool
        elif config.pooling == 'mean':
            self.pool = global_mean_pool

        self.dropout = config.dropout

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.lin0(x)
        xs = []
        for conv in self.convs:
            x = conv(x, edge_index)
            xs += [x]

        x = self.JK(xs)

        x = self.pool(x, batch)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
