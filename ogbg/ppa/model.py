import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, ELU, Sigmoid
from utils.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn import global_add_pool, global_mean_pool
from conv import GinConv, ExpandingBConv, CombBConv, ExpandingAConv, CombAConv


class Net(torch.nn.Module):
    def __init__(self,
                 config,
                 num_class):
        super(Net, self).__init__()
        self.node_encoder = torch.nn.Embedding(1, config.hidden)

        self.convs = torch.nn.ModuleList()
        if config.nonlinear_conv == 'GIN':
            for i in range(config.layers):
                self.convs.append(GinConv(config.hidden, config.variants))
        elif config.nonlinear_conv[:2] == 'EB':
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
        else:
            ValueError('Undefined conv called {}'.format(config.nonlinear_conv))

        self.JK = JumpingKnowledge(config.JK)

        if config.JK == 'cat':
            self.graph_pred_linear = torch.nn.Linear(config.hidden * config.layers, num_class)
        else:
            self.graph_pred_linear = torch.nn.Linear(config.hidden, num_class)

        if config.pooling == 'add':
            self.pool = global_add_pool
        elif config.pooling == 'mean':
            self.pool = global_mean_pool

        self.dropout = config.dropout

    def reset_parameters(self):
        self.atom_encoder.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.graph_pred_linear.reset_parameters()

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        x = self.node_encoder(x)
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
