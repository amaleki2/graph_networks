import torch.nn
import torch
import torch.nn
from torch.nn import Sequential, ReLU, Linear, LayerNorm, LazyLinear
from torch_scatter import scatter_sum

__all__ = ['make_lazy_mlp_model',
           'cast_globals_to_nodes',
           'cast_edges_to_nodes',
           'cast_globals_to_edges',
           'cast_edges_to_globals',
           'cast_nodes_to_globals']


def make_mlp_model(n_input, latent_sizes, n_output, activate_final=False, normalize=True):
    if latent_sizes is None:
        mlp_sizes = [n_input, n_output]
    elif isinstance(latent_sizes, int):
        mlp_sizes = [n_input, latent_sizes, n_output]
    elif isinstance(latent_sizes, list):
        mlp_sizes = [n_input] + latent_sizes + [n_output]
    else:
        raise ValueError

    mlp = []
    for i in range(len(mlp_sizes) - 1):
        mlp.append(Linear(mlp_sizes[i], mlp_sizes[i+1]))
        if i < len(mlp_sizes) - 2 or activate_final:
            mlp.append(ReLU(inplace=True))

    if normalize and latent_sizes is not None:
        mlp.append(LayerNorm(n_output))

    mlp = Sequential(*mlp)

    return mlp


def make_lazy_mlp_model(latent_sizes, n_output, activate_final=False, normalize=True):
    if latent_sizes is None:
        mlp_sizes = [n_output]
    elif isinstance(latent_sizes, int):
        mlp_sizes = [latent_sizes, n_output]
    elif isinstance(latent_sizes, list):
        mlp_sizes = latent_sizes + [n_output]
    else:
        raise ValueError

    lazy_mlp = [LazyLinear(mlp_sizes[0])]
    for i in range(len(mlp_sizes) - 1):
        lazy_mlp.append(Linear(mlp_sizes[i], mlp_sizes[i + 1]))
        if i < len(mlp_sizes) - 2 or activate_final:
            lazy_mlp.append(ReLU(inplace=True))

    if normalize and latent_sizes is not None:
        lazy_mlp.append(LayerNorm(n_output))

    mlp = Sequential(*lazy_mlp)

    return mlp


def get_edge_counts(edge_index, batch):
    return torch.bincount(batch[edge_index[0, :]])


def cast_globals_to_nodes(global_attr, batch=None, num_nodes=None):
    if batch is not None:
        _, counts = torch.unique(batch, return_counts=True)
        casted_global_attr = torch.cat([torch.repeat_interleave(global_attr[idx:idx+1, :], rep, dim=0)
                                        for idx, rep in enumerate(counts)], dim=0)
    else:
        assert global_attr.size(0) == 1, "batch numbers should be provided."
        assert num_nodes is not None, "number of nodes should be specified."
        casted_global_attr = torch.cat([global_attr] * num_nodes, dim=0)
    return casted_global_attr


def cast_globals_to_edges(global_attr, edge_index=None, batch=None, num_edges=None):
    if batch is not None:
        assert edge_index is not None, "edge index should be specified"
        edge_counts = get_edge_counts(edge_index, batch)
        casted_global_attr = torch.cat([torch.repeat_interleave(global_attr[idx:idx+1, :], rep, dim=0)
                                        for idx, rep in enumerate(edge_counts)], dim=0)
    else:
        assert global_attr.size(0) == 1, "batch numbers should be provided."
        assert num_edges is not None, "number of edges should be specified"
        casted_global_attr = torch.cat([global_attr] * num_edges, dim=0)
    return casted_global_attr


def cast_edges_to_globals(edge_attr, edge_index=None, batch=None, num_edges=None, num_globals=None):
    if batch is None:
        edge_attr_aggr = torch.sum(edge_attr, dim=0, keepdim=True)
    else:
        node_indices = torch.unique(batch)
        edge_counts = get_edge_counts(edge_index, batch)
        assert sum(edge_counts) == num_edges
        # indices = [idx.view(1, 1) for idx, count in zip(node_indices, edge_counts) for _ in range(count)]
        indices = [torch.repeat_interleave(idx, count) for idx, count in zip(node_indices, edge_counts)]
        indices = torch.cat(indices)
        edge_attr_aggr = scatter_sum(edge_attr, index=indices, dim=0, dim_size=num_globals)
    return edge_attr_aggr


def cast_nodes_to_globals(node_attr, batch=None, num_globals=None):
    if batch is None:
        x_aggr = torch.sum(node_attr, dim=0, keepdim=True)
    else:
        x_aggr = scatter_sum(node_attr, index=batch, dim=0, dim_size=num_globals)
    return x_aggr


def cast_edges_to_nodes(edge_attr, indices, num_nodes=None):
    edge_attr_aggr = scatter_sum(edge_attr, indices, dim=0, dim_size=num_nodes)
    return edge_attr_aggr