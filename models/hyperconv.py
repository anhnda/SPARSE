import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import uniform


class HyperConv(MessagePassing):
    r"""
    Implementation of message passing for clique expansion of hypergraphs.
    n_type = 5 is for 5 types of propagation:
    0 for drug-drug (two different drugs),
    1 for drug-side effect,
    2 for side effect-drug ,
    3 for drug self-loop,
    4 for side-effect self-loop
    """
    def __init__(self, in_channels, out_channels, normalize=False,
                 concat=False, bias=True, skip_last_weight=True, n_type=5, device=None, **kwargs):
        super(HyperConv, self).__init__(aggr='mean', **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.concat = concat
        self.n_type = n_type
        self.device = device
        in_channels = 2 * in_channels if concat else in_channels
        self.transforms = [torch.nn.Linear(in_channels, out_channels, bias=True).to(device) for _ in range(n_type)]
        self.skip_last_weight = skip_last_weight
        self.weight = Parameter(torch.Tensor(in_channels, out_channels).to(device))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels).to(device))
        else:
            self.register_parameter('bias', None)
        self.register_new_params()
        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.weight.size(0), self.weight)
        uniform(self.weight.size(0), self.bias)
        for transform in self.transforms:
            uniform(transform.weight.size(1), transform.weight)

    def register_new_params(self):
        for i, transform in enumerate(self.transforms):
            self.register_parameter("transform_%s" % i, transform.weight)
            self.register_parameter("bias_%s" % i, transform.bias)

    def forward(self, x, edge_index, edge_types):
        return self.propagate(edge_index, x=x, edge_types=edge_types)

    def message(self, x_j):
        return x_j

    def update(self, aggr_out, x):
        if self.skip_last_weight:
            return aggr_out

        if self.concat and torch.is_tensor(x):
            aggr_out = torch.cat([x, aggr_out], dim=-1)

        aggr_out = torch.matmul(aggr_out, self.weight)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias

        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)

        return aggr_out

    def __collect__(self, edge_index, edge_types, x):
        r"""
        Collecting information passing through source-target of each pair of nodes
        Args:
            edge_index: (2 x num_pairs) Tensor for the source to target node
            edge_types: either 0, 1, 2, 3, 4
            x: Input embeddings of nodes

        Returns:
            Embeddings after propagations through each pairs
        """
        n_out = edge_index.size(1)
        x_out = torch.empty((n_out, self.out_channels), device=self.device)
        source_ids = edge_index[0, :]
        for i in range(self.n_type):
            ids = torch.nonzero(edge_types == i).to(self.device)
            x_ids = x[source_ids[ids]]
            x_out_v = self.transforms[i](x_ids)
            x_out[ids] = x_out_v
        return x_out

    def propagate(self, edge_index, edge_types, x):

        mp_type = self.__get_mp_type__(edge_index)
        assert mp_type == 'edge_index'
        x_out = self.__collect__(edge_index, edge_types, x)
        out = self.message(x_out)
        out = self.aggregate(out, edge_index[1, :])
        out = self.update(out, x)

        return out

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)
