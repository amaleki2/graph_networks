"""
Test the Box object
"""
import os
import json
import torch
import unittest
from src.graph_networks import EncodeProcessDecode
from torch_geometric.data import Data, DataLoader



class GraphEPDTest(unittest.TestCase):
    @staticmethod
    def get_abs_path():
        path = os.path.abspath(__file__)
        parent_dir = os.path.split(path)[0]
        return parent_dir

    def get_random_pyg_data(self, n_data=2, get_first_element=True):
        graph_data = [Data(x=torch.rand(3, 4),
                           e=torch.rand(5, 3),
                           u=torch.rand(1, 2),
                           edge_index=torch.randint(3, (2, 5))) for _ in range(n_data)]
        data = DataLoader(graph_data, batch_size=2, shuffle=True)
        if get_first_element:
            data = next(iter(data))
        return data

    def test_epd(self):
        data1 = self.get_random_pyg_data()
        parent_dir = self.get_abs_path()
        config_file = os.path.join(parent_dir, "test_configs.json")
        with open(config_file, "rb") as fid:
            config = json.load(fid)

        model_params = config['encode_process_decode']
        model1 = EncodeProcessDecode(**model_params)

        model1(data1)

        model_params["decoder_params"]["edge_model_params"] = None

        model2 = EncodeProcessDecode(**model_params)
        model2(data1)


if __name__ == '__main__':
    unittest.main()