# Taken from https://github.com/vgsatorras/en_flows/blob/main/egnn/models.py

import torch
import torch.nn as nn


def remove_mean(x):
    mean = torch.mean(x, dim=1, keepdim=True)
    x = x - mean
    return x


def log_map(X, Y, L):
    diff = Y - X
    return diff - torch.round(diff / L) * L


def exp_map(X, Y, L):
    return torch.remainder(X + Y, L)


def dist_sq_torus(X, Y, L):
    return torch.sum(torch.square(log_map(X, Y, L)), dim=-1)


class EGNN_dynamics(nn.Module):
    def __init__(
        self,
        n_particles,
        n_dimension,
        hidden_nf=64,
        act_fn=torch.nn.SiLU(),
        n_layers=4,
        recurrent=True,
        attention=False,
        condition_time=True,
        tanh=False,
        agg="sum",
        L=None,
        n_species=1,
    ):
        super().__init__()
        self.egnn = EGNN(
            in_node_nf=1+n_species if n_species > 1 else 1,
            in_edge_nf=1,
            hidden_nf=hidden_nf,
            act_fn=act_fn,
            n_layers=n_layers,
            recurrent=recurrent,
            attention=attention,
            tanh=tanh,
            agg=agg,
            L=L
        )

        self._n_particles = n_particles
        self._n_dimension = n_dimension
        self.edges = self._create_edges()
        self._edges_dict = {}
        self.condition_time = condition_time
        self.L = L
        self.n_species = n_species
        if self.n_species > 1:
            self.species_embedding = torch.nn.Embedding(self.n_species, self.n_species,
                                                        sparse=True, _weight=torch.eye(self.n_species), _freeze=True)

    def forward(self, t, xs, a=None):

        n_batch = xs.shape[0]
        edges = self._cast_edges2batch(self.edges, n_batch, self._n_particles)
        edges = [edges[0], edges[1]]
        x = xs.view(n_batch * self._n_particles, self._n_dimension)
        if self.condition_time:
            if len(t.shape) == 0:
                h = t * torch.ones(n_batch * self._n_particles, 1, device=xs.device)
            else:
                h = t.expand((-1, self._n_particles, -1)).reshape((-1, 1))
        else:
            h = torch.ones(n_batch * self._n_particles, 1, device=xs.device)
        if self.n_species > 1:
            if a is None:
                raise ValueError('Must provide a when n_species > 1.')
            h = torch.concatenate([
                h, self.species_embedding(a).view((n_batch * self._n_particles, -1))
            ], dim=-1)
        if self.L is not None:
            edge_attr = dist_sq_torus(x[edges[0]], x[edges[1]], self.L).unsqueeze(-1)
        else:
            edge_attr = torch.sum((x[edges[0]] - x[edges[1]]) ** 2, dim=1, keepdim=True)
        _, x_final = self.egnn(h, x.clone(), edges, edge_attr=edge_attr)
        if self.L is not None:
            vel = log_map(x, x_final, self.L)
        else:
            vel = x_final - x
        vel = vel.view(n_batch, self._n_particles, self._n_dimension)
        if self.L is None:
            vel = remove_mean(vel)
        return vel

    def _create_edges(self):
        rows, cols = [], []
        for i in range(self._n_particles):
            for j in range(i + 1, self._n_particles):
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
        return [torch.LongTensor(rows), torch.LongTensor(cols)]

    def _cast_edges2batch(self, edges, n_batch, n_nodes):
        if n_batch not in self._edges_dict:
            self._edges_dict = {}
            rows, cols = edges
            rows_total, cols_total = [], []
            for i in range(n_batch):
                rows_total.append(rows + i * n_nodes)
                cols_total.append(cols + i * n_nodes)
            rows_total = torch.cat(rows_total)
            cols_total = torch.cat(cols_total)

            self._edges_dict[n_batch] = [rows_total, cols_total]
        return self._edges_dict[n_batch]

    def _apply(self, fn):
        new_self = super(EGNN_dynamics, self)._apply(fn)
        new_self.edges[0] = fn(new_self.edges[0])
        new_self.edges[1] = fn(new_self.edges[1])
        for k in new_self._edges_dict.keys():
            new_self._edges_dict[k][0] = fn(new_self._edges_dict[k][0])
            new_self._edges_dict[k][1] = fn(new_self._edges_dict[k][1])
        return new_self


class EGNN(nn.Module):
    def __init__(
        self,
        in_node_nf,
        in_edge_nf,
        hidden_nf,
        act_fn=nn.SiLU(),
        n_layers=4,
        recurrent=True,
        attention=False,
        norm_diff=True,
        out_node_nf=None,
        tanh=False,
        coords_range=15,
        agg="sum",
        L=None
    ):
        super(EGNN, self).__init__()
        if out_node_nf is None:
            out_node_nf = in_node_nf
        self.hidden_nf = hidden_nf
        self.n_layers = n_layers
        self.coords_range_layer = float(coords_range) / self.n_layers
        if agg == "mean":
            self.coords_range_layer = self.coords_range_layer * 19
        # self.reg = reg
        # Encoder
        # self.add_module("gcl_0", E_GCL(in_node_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, act_fn=act_fn, recurrent=False, coords_weight=coords_weight))
        self.embedding = nn.Linear(in_node_nf, self.hidden_nf)
        self.embedding_out = nn.Linear(self.hidden_nf, out_node_nf)
        for i in range(0, n_layers):
            self.add_module(
                "gcl_%d" % i,
                E_GCL(
                    self.hidden_nf,
                    self.hidden_nf,
                    self.hidden_nf,
                    edges_in_d=in_edge_nf,
                    act_fn=act_fn,
                    recurrent=recurrent,
                    attention=attention,
                    norm_diff=norm_diff,
                    tanh=tanh,
                    coords_range=self.coords_range_layer,
                    agg=agg,
                    L=L
                ),
            )

    def forward(self, h, x, edges, edge_attr=None, node_mask=None, edge_mask=None):
        # Edit Emiel: Remove velocity as input
        h = self.embedding(h)
        for i in range(0, self.n_layers):
            h, x, _ = self._modules["gcl_%d" % i](
                h,
                edges,
                x,
                edge_attr=edge_attr,
                node_mask=node_mask,
                edge_mask=edge_mask,
            )
        h = self.embedding_out(h)

        # Important, the bias of the last linear might be non-zero
        if node_mask is not None:
            h = h * node_mask
        return h, x


class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(
        self,
        input_nf,
        output_nf,
        hidden_nf,
        edges_in_d=0,
        nodes_att_dim=0,
        act_fn=nn.SiLU(),
        recurrent=True,
        attention=False,
        clamp=False,
        norm_diff=True,
        tanh=False,
        coords_range=1,
        agg="sum",
        L=None
    ):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.agg_type = agg
        self.tanh = tanh
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf),
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = coords_range

        self.coord_mlp = nn.Sequential(*coord_mlp)
        self.clamp = clamp

        if self.attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

        # if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)

        self.L = L
        if (agg != 'sum') and (self.L is not None):
            raise ValueError("Only agg = 'sum' supported when L is not None.")

    def edge_model(self, source, target, radial, edge_attr, edge_mask):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val

        if edge_mask is not None:
            out = out * edge_mask
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(
        self, coord, edge_index, coord_diff, radial, edge_feat, node_mask, edge_mask
    ):
        row, col = edge_index
        if self.tanh:
            trans = coord_diff * self.coord_mlp(edge_feat) * self.coords_range
        else:
            trans = coord_diff * self.coord_mlp(edge_feat)
        # trans = torch.clamp(trans, min=-100, max=100)
        if edge_mask is not None:
            trans = trans * edge_mask

        if self.agg_type == "sum":
            agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0), out=coord)
        elif self.agg_type == "mean":
            if node_mask is not None:
                # raise Exception('This part must be debugged before use')
                agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
                M = unsorted_segment_sum(node_mask[col], row, num_segments=coord.size(0))
                coord += agg / (M - 1)
            else:
                unsorted_segment_mean(trans, row, num_segments=coord.size(0), out=coord)
        else:
            raise Exception("Wrong coordinates aggregation type")
        if self.L is not None:
            coord = torch.remainder(coord, self.L)
        return coord

    def forward(
        self,
        h,
        edge_index,
        coord,
        edge_attr=None,
        node_attr=None,
        node_mask=None,
        edge_mask=None,
    ):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr, edge_mask)
        coord = self.coord_model(
            coord, edge_index, coord_diff, radial, edge_feat, node_mask, edge_mask
        )

        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN

        if node_mask is not None:
            h = h * node_mask
            coord = coord * node_mask
        return h, coord, edge_attr

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        if self.L is not None:
            coord_diff = log_map(coord[col], coord[row], self.L)
        else:
            coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff) ** 2, 1).unsqueeze(1)  # the norm is right for the torus
        norm = torch.sqrt(radial + 1e-8)
        coord_diff = coord_diff / (norm + 1)
        return radial, coord_diff


def broadcast(src, other, dim):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src

def scatter_sum(src, index, dim=-1, out=None, dim_size=None, L=None):
    """Imported from torch_scatter"""
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        # return out.scatter_add_(dim, index, src)
        if L is not None:
            return torch.remainder(torch.scatter_add(out, dim, index, src), L)
        else:
            return torch.scatter_add(out, dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)

def scatter_mean(src, index, dim=-1, out=None, dim_size=None):
    """Imported from torch_scatter"""
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)
    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1
    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode='floor')
    return out

def unsorted_segment_sum(data, segment_ids, num_segments, out=None, L=None):
    return scatter_sum(data, segment_ids, dim=0, dim_size=num_segments, out=out, L=L)

def unsorted_segment_mean(data, segment_ids, num_segments, out=None):
    return scatter_mean(data, segment_ids, dim=0, dim_size=num_segments, out=out)
