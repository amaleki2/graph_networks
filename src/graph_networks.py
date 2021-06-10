import torch
from .modules import GraphNetwork


class EncodeProcessDecode(torch.nn.Module):
    def __init__(self,
                 encoder_params=None,
                 processor_params=None,
                 decoder_params=None,
                 outputer_params=None,
                 num_processing_steps=2):
        super(EncodeProcessDecode, self).__init__()
        self.num_processing_steps = num_processing_steps
        self.encoder   = GraphNetwork(**encoder_params)
        self.processor = GraphNetwork(**processor_params)
        self.decoder   = GraphNetwork(**decoder_params)
        self.outputer  = GraphNetwork(**outputer_params) if outputer_params is not None else None

    def forward(self, graph):
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


