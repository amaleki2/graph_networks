import torch
from .utils import *


INCLUDE_GLOBALS_DEFAULT = True
INCLUDE_SENDERS_DEFAULT = False


class BaseModel(torch.nn.Module):
    def __init__(self,
                 n_feat_out=16,
                 latent_sizes=16,
                 activate_final=True,
                 normalize=True,
                 independent=False,
                 **kwargs
                 ):
        super(BaseModel, self).__init__()
        self.independent = independent
        self.kwargs = kwargs
        self.mlp = make_lazy_mlp_model(latent_sizes,
                                       n_feat_out,
                                       activate_final=activate_final,
                                       normalize=normalize)

    def collect_attrs(self, graph):
        raise NotImplementedError

    def forward(self, graph, concat_graph=None):
        attrs = self.collect_attrs(graph)
        if concat_graph is not None:
            concat_attrs = self.collect_attrs(concat_graph)
            attrs = [torch.cat((attr, concat_attr), dim=-1) for (attr, concat_attr) in zip(attrs, concat_attrs)]

        atrrs = torch.cat(attrs, dim=1)
        return self.mlp(atrrs)


class EdgeModel(BaseModel):
    def collect_attrs(self, graph):
        if self.independent:
            return [graph.e]

        node_attr, edge_attr, global_attr, edge_index, batch = graph.x, graph.e, graph.u, graph.edge_index, graph.batch
        num_nodes, num_edges, num_globals = node_attr.size(0), edge_attr.size(0), global_attr.size(0)
        row, col = edge_index
        sender_attr, receiver_attr = node_attr[row, :], node_attr[col, :]
        out = [edge_attr, receiver_attr, sender_attr]

        with_globals = self.kwargs.get('with_globals', INCLUDE_GLOBALS_DEFAULT)
        if with_globals:
            global_attr_scattered = cast_globals_to_edges(global_attr, edge_index=edge_index, batch=batch,
                                                          num_edges=num_edges)
            out.append(global_attr_scattered)
        return out

    def forward(self, graph, concat_graph=None):
        graph.e = super().forward(graph, concat_graph=concat_graph)
        return graph


class NodeModel(BaseModel):
    def collect_attrs(self, graph):
        if self.independent:
            return [graph.x]

        node_attr, edge_attr, global_attr, edge_index, batch = graph.x, graph.e, graph.u, graph.edge_index, graph.batch
        num_nodes, num_edges, num_globals = node_attr.size(0), edge_attr.size(0), global_attr.size(0)
        row, col = edge_index
        receiver_attr_to_node = cast_edges_to_nodes(edge_attr, col, num_nodes=num_nodes)
        out = [node_attr, receiver_attr_to_node]

        with_senders = self.kwargs.get('with_senders', INCLUDE_SENDERS_DEFAULT)
        if with_senders:
            sender_attr_to_node = cast_edges_to_nodes(edge_attr, row, num_nodes=num_nodes)
            out.append(sender_attr_to_node)

        with_globals = self.kwargs.get('with_globals', INCLUDE_GLOBALS_DEFAULT)
        if with_globals:
            global_attr_to_nodes = cast_globals_to_nodes(global_attr, batch=batch, num_nodes=num_nodes)
            out.append(global_attr_to_nodes)

        return out

    def forward(self, graph, concat_graph=None):
        graph.x = super().forward(graph, concat_graph=concat_graph)
        return graph


class GlobalModel(BaseModel):
    def collect_attrs(self, graph):
        if self.independent:
            return [graph.u]

        node_attr, edge_attr, global_attr, edge_index, batch = graph.x, graph.e, graph.u, graph.edge_index, graph.batch
        num_edges = edge_attr.size(0)
        num_globals = global_attr.size(0)

        node_attr_aggr = cast_nodes_to_globals(node_attr, batch=batch, num_globals=num_globals)
        edge_attr_aggr = cast_edges_to_globals(edge_attr, edge_index=edge_index, batch=batch,
                                               num_edges=num_edges, num_globals=num_globals)
        out = [global_attr, node_attr_aggr, edge_attr_aggr]
        return out

    def forward(self, graph, concat_graph=None):
        graph.u = super().forward(graph, concat_graph=concat_graph)
        return graph


class GraphNetwork(torch.nn.Module):
    def __init__(self,
                 edge_model_params=None,
                 node_model_params=None,
                 global_model_params=None):
        super(GraphNetwork, self).__init__()

        if edge_model_params is None:
            self.edge_model = None
        else:
            self.edge_model = EdgeModel(**edge_model_params)

        if node_model_params is None:
            self.node_model = None
        else:
            self.node_model = NodeModel(**node_model_params)

        if global_model_params is None:
            self.global_model = None
        else:
            self.global_model = GlobalModel(**global_model_params)

    def forward(self, graph, concat_graph=None):
        graph = self.edge_model(graph, concat_graph=concat_graph) if self.edge_model is not None else graph
        graph = self.node_model(graph, concat_graph=concat_graph) if self.node_model is not None else graph
        graph = self.global_model(graph, concat_graph=concat_graph) if self.global_model is not None else graph
        return graph

