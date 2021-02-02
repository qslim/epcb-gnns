import torch
import torch.nn.functional as F
from utils.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn import global_add_pool, global_mean_pool
from conv import GinConv, ExpC, CombC, ExpC_star, CombC_star
from ogb.graphproppred.mol_encoder import AtomEncoder


class Net(torch.nn.Module):
    def __init__(self,
                 config,
                 num_tasks):
        super(Net, self).__init__()
        self.atom_encoder = AtomEncoder(config.hidden)

        self.convs = torch.nn.ModuleList()
        if config.methods == 'GIN':
            for i in range(config.layers):
                self.convs.append(GinConv(config.hidden, config.variants))
        elif config.methods[:2] == 'EB':
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
            self.graph_pred_linear = torch.nn.Linear(config.hidden * config.layers, num_tasks)
        else:
            self.graph_pred_linear = torch.nn.Linear(config.hidden, num_tasks)

        if config.pooling == 'add':
            self.pool = global_add_pool
        elif config.pooling == 'mean':
            self.pool = global_mean_pool

        self.dropout = config.dropout

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        x = self.atom_encoder(x)
        xs = []
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            xs += [x]

        nr = self.JK(xs)

        nr = F.dropout(nr, p=self.dropout, training=self.training)
        h_graph = self.pool(nr, batched_data.batch)
        return self.graph_pred_linear(h_graph)

    def __repr__(self):
        return self.__class__.__name__
