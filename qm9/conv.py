import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops
from torch.nn import Linear, Sequential, Tanh, ReLU, ELU, BatchNorm1d as BN


class ExpC(MessagePassing):
    def __init__(self, hidden, num_aggr, config, **kwargs):
        super(ExpC, self).__init__(aggr='add', **kwargs)
        self.hidden = hidden
        self.num_aggr = num_aggr

        if config.fea_activation == 'ELU':
            self.fea_activation = ELU()
        elif config.fea_activation == 'ReLU':
            self.fea_activation = ReLU()

        self.fea_mlp = Sequential(
            Linear(hidden * self.num_aggr, hidden),
            ReLU(),
            Linear(hidden, hidden),
            self.fea_activation)

        self.aggr_mlp = Sequential(
            Linear(hidden * 2, self.num_aggr),
            Tanh())

        self.edge_encoder = torch.nn.Linear(5, hidden)

        if config.BN == 'Y':
            self.BN = BN(hidden)
        else:
            self.BN = None

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.edge_encoder(edge_attr)
        edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        if self.BN is not None:
            out = self.BN(out)
        return out

    def message(self, x_i, x_j, edge_attr):
        xe = x_j + edge_attr
        aggr_emb = self.aggr_mlp(torch.cat([x_i, xe], dim=-1))
        feature2d = torch.matmul(aggr_emb.unsqueeze(-1), xe.unsqueeze(-1)
                                 .transpose(-1, -2)).squeeze(-1).view(-1, self.hidden * self.num_aggr)
        return self.fea_mlp(feature2d)

    def update(self, aggr_out, x):
        root_emb = self.aggr_mlp(torch.cat([x, x], dim=-1))
        feature2d = torch.matmul(root_emb.unsqueeze(-1), x.unsqueeze(-1)
                                 .transpose(-1, -2)).squeeze(-1).view(-1, self.hidden * self.num_aggr)
        return aggr_out + self.fea_mlp(feature2d)

    def __repr__(self):
        return self.__class__.__name__


class ExpC_star(MessagePassing):
    def __init__(self, hidden, num_aggr, config, **kwargs):
        super(ExpC_star, self).__init__(aggr='add', **kwargs)
        self.hidden = hidden
        self.num_aggr = num_aggr

        if config.fea_activation == 'ELU':
            self.fea_activation = ELU()
        elif config.fea_activation == 'ReLU':
            self.fea_activation = ReLU()

        self.fea_mlp = Sequential(
            Linear(hidden * self.num_aggr, hidden),
            ReLU(),
            Linear(hidden, hidden),
            self.fea_activation)

        self.aggr_mlp = Sequential(
            Linear(hidden * 2, self.num_aggr),
            Tanh())

        self.edge_encoder = torch.nn.Linear(5, hidden)

        if config.BN == 'Y':
            self.BN = BN(hidden)
        else:
            self.BN = None

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.edge_encoder(edge_attr)
        edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.fea_mlp(self.propagate(edge_index, x=x, edge_attr=edge_attr))
        if self.BN is not None:
            out = self.BN(out)
        return out

    def message(self, x_i, x_j, edge_attr):
        xe = x_j + edge_attr
        aggr_emb = self.aggr_mlp(torch.cat([x_i, xe], dim=-1))
        feature2d = torch.matmul(aggr_emb.unsqueeze(-1), xe.unsqueeze(-1)
                                 .transpose(-1, -2)).squeeze(-1).view(-1, self.hidden * self.num_aggr)
        return feature2d

    def update(self, aggr_out, x):
        root_emb = self.aggr_mlp(torch.cat([x, x], dim=-1))
        feature2d = torch.matmul(root_emb.unsqueeze(-1), x.unsqueeze(-1)
                                 .transpose(-1, -2)).squeeze(-1).view(-1, self.hidden * self.num_aggr)
        return aggr_out + feature2d

    def __repr__(self):
        return self.__class__.__name__


class CombC(MessagePassing):
    def __init__(self, hidden, config, **kwargs):
        super(CombC, self).__init__(aggr='add', **kwargs)

        if config.fea_activation == 'ELU':
            self.fea_activation = ELU()
        elif config.fea_activation == 'ReLU':
            self.fea_activation = ReLU()

        self.fea_mlp = Sequential(
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, hidden),
            self.fea_activation)

        self.aggr_mlp = Sequential(
            Linear(hidden * 2, hidden),
            Tanh())

        self.edge_encoder = torch.nn.Linear(5, hidden)

        if config.BN == 'Y':
            self.BN = BN(hidden)
        else:
            self.BN = None

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.edge_encoder(edge_attr)
        edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        if self.BN is not None:
            out = self.BN(out)
        return out

    def message(self, x_i, x_j, edge_attr):
        xe = x_j + edge_attr
        aggr_emb = self.aggr_mlp(torch.cat([x_i, xe], dim=-1))
        return self.fea_mlp(aggr_emb * xe)

    def update(self, aggr_out, x):
        root_emb = self.aggr_mlp(torch.cat([x, x], dim=-1))
        return aggr_out + self.fea_mlp(root_emb * x)

    def __repr__(self):
        return self.__class__.__name__


class CombC_star(MessagePassing):
    def __init__(self, hidden, config, **kwargs):
        super(CombC_star, self).__init__(aggr='add', **kwargs)

        if config.fea_activation == 'ELU':
            self.fea_activation = ELU()
        elif config.fea_activation == 'ReLU':
            self.fea_activation = ReLU()

        self.fea_mlp = Sequential(
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, hidden),
            self.fea_activation)

        self.aggr_mlp = Sequential(
            Linear(hidden * 2, hidden),
            Tanh())

        self.edge_encoder = torch.nn.Linear(5, hidden)

        if config.BN == 'Y':
            self.BN = BN(hidden)
        else:
            self.BN = None

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.edge_encoder(edge_attr)
        edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.fea_mlp(self.propagate(edge_index, x=x, edge_attr=edge_attr))
        if self.BN is not None:
            out = self.BN(out)
        return out

    def message(self, x_i, x_j, edge_attr):
        xe = x_j + edge_attr
        aggr_emb = self.aggr_mlp(torch.cat([x_i, xe], dim=-1))
        return aggr_emb * xe

    def update(self, aggr_out, x):
        root_emb = self.aggr_mlp(torch.cat([x, x], dim=-1))
        return aggr_out + root_emb * x

    def __repr__(self):
        return self.__class__.__name__


class GinConv(MessagePassing):
    def __init__(self, hidden, config, **kwargs):
        super(GinConv, self).__init__(aggr='add', **kwargs)

        if config.fea_activation == 'ELU':
            self.fea_activation = ELU()
        elif config.fea_activation == 'ReLU':
            self.fea_activation = ReLU()

        self.fea_mlp = Sequential(
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, hidden),
            self.fea_activation)

        self.edge_encoder = torch.nn.Linear(5, hidden)

        if config.BN == 'Y':
            self.BN = BN(hidden)
        else:
            self.BN = None

    def forward(self, x, edge_index, edge_attr):
        edge_attr = self.edge_encoder(edge_attr)
        edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.fea_mlp(self.propagate(edge_index, x=x, edge_attr=edge_attr))
        if self.BN is not None:
            out = self.BN(out)
        return out

    def message(self, x_j, edge_attr):
        return x_j + edge_attr

    def update(self, aggr_out, x):
        return aggr_out + x

    def __repr__(self):
        return self.__class__.__name__
