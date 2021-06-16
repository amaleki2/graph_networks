import torch
from .modules import GraphNetwork
from torch_geometric.nn import GATConv


class GAT(GATConv):
    def forward(self, graph, concat_graph=None, **kwargs):
        x = graph.x if concat_graph is None else torch.cat((graph.x, concat_graph.x), dim=-1)
        graph.x = super().forward(x, graph.edge_index, **kwargs)
        return graph


class EncodeProcessDecode(torch.nn.Module):
    graph_classes = {'GraphNetwork': GraphNetwork, 'GATConv': GAT}
    def __init__(self,
                 encoder_params=None,
                 processor_params=None,
                 decoder_params=None,
                 outputer_params=None,
                 num_processing_steps=2):
        super(EncodeProcessDecode, self).__init__()

        self.num_processing_steps = num_processing_steps
        self.encoder   = self._setup_graph_from_params(encoder_params)
        self.processor = self._setup_graph_from_params(processor_params)
        self.decoder   = self._setup_graph_from_params(decoder_params)
        self.outputer  = self._setup_graph_from_params(outputer_params)

    def _setup_graph_from_params(self, params):
        if params is None:
            return None
        else:
            graph_type = params.pop("graph_type", "GraphNetwork")
            return self.graph_classes[graph_type](**params)

    def forward(self, graph0):
        graph = graph0.clone()  # dont change the input
        graph = self.encoder(graph)
        encoded = graph.clone()
        for i in range(self.num_processing_steps):
            graph = self.processor(graph, concat_graph=encoded)
        decoded = self.decoder(graph)
        if self.outputer is None:
            out = decoded
        else:
            out = self.outputer(decoded)
        return out

