import torch
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax
import torch.nn.functional as F
from torch.nn import Linear, Sequential, Tanh, ReLU, ELU, BatchNorm1d as BN, Parameter


class ExpC(MessagePassing):
    def __init__(self, hidden, num_aggr, config, **kwargs):
        super(ExpC, self).__init__(aggr='add', **kwargs)
        self.hidden = hidden
        self.num_aggr = num_aggr

        self.fea_mlp = Sequential(
            Linear(hidden * self.num_aggr, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU())

        self.aggr_mlp = Sequential(
            Linear(hidden * 2, self.num_aggr),
            Tanh())

        if config.BN == 'Y':
            self.BN = BN(hidden)
        else:
            self.BN = None

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x)
        if self.BN is not None:
            out = self.BN(out)
        return out

    def message(self, x_i, x_j):
        xe = x_j
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

        self.fea_mlp = Sequential(
            Linear(hidden * self.num_aggr, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU())

        self.aggr_mlp = Sequential(
            Linear(hidden * 2, self.num_aggr),
            Tanh())

        if config.BN == 'Y':
            self.BN = BN(hidden)
        else:
            self.BN = None

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.fea_mlp(self.propagate(edge_index, x=x))
        if self.BN is not None:
            out = self.BN(out)
        return out

    def message(self, x_i, x_j):
        xe = x_j
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

        self.fea_mlp = Sequential(
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU())

        self.aggr_mlp = Sequential(
            Linear(hidden * 2, hidden),
            Tanh())

        if config.BN == 'Y':
            self.BN = BN(hidden)
        else:
            self.BN = None

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.propagate(edge_index, x=x)
        if self.BN is not None:
            out = self.BN(out)
        return out

    def message(self, x_i, x_j):
        xe = x_j
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

        self.fea_mlp = Sequential(
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU())

        self.aggr_mlp = Sequential(
            Linear(hidden * 2, hidden),
            Tanh())

        if config.BN == 'Y':
            self.BN = BN(hidden)
        else:
            self.BN = None

    def forward(self, x, edge_index):
        edge_index, _ = remove_self_loops(edge_index)
        # edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        out = self.fea_mlp(self.propagate(edge_index, x=x))
        if self.BN is not None:
            out = self.BN(out)
        return out

    def message(self, x_i, x_j):
        xe = x_j
        aggr_emb = self.aggr_mlp(torch.cat([x_i, xe], dim=-1))
        return aggr_emb * xe

    def update(self, aggr_out, x):
        root_emb = self.aggr_mlp(torch.cat([x, x], dim=-1))
        return aggr_out + root_emb * x

    def __repr__(self):
        return self.__class__.__name__
