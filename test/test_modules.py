import torch
import unittest
from src.modules import EdgeModel, GraphNetwork, NodeModel, GlobalModel
from torch_geometric.data import Data, DataLoader


class GraphModulesTest(unittest.TestCase):
    def get_random_pyg_data(self, n_data=2, get_first_element=True):
        graph_data = [Data(x=torch.rand(3, 4),
                           e=torch.rand(5, 3),
                           u=torch.rand(1, 2),
                           edge_index=torch.randint(3, (2, 5))) for _ in range(n_data)]
        data = DataLoader(graph_data, batch_size=2, shuffle=True)
        if get_first_element:
            data = next(iter(data))
        return data

    def test_edge_model(self):
        data = self.get_random_pyg_data()
        model1 = EdgeModel(n_feat_out=16,
                           latent_sizes=16,
                           activate_final=True,
                           normalize=False,
                           independent=False)
        model1(data)

        model2 = EdgeModel(n_feat_out=16,
                           latent_sizes=16,
                           activate_final=True,
                           with_globals=False,
                           normalize=True,
                           independent=False)
        model2(data)

        model3 = EdgeModel(n_feat_out=16,
                           latent_sizes=16,
                           activate_final=True,
                           with_globals=False,
                           normalize=True,
                           independent=True)
        model3(data)

    def test_node_model(self):
        data = self.get_random_pyg_data()
        model1 = NodeModel(n_feat_out=16,
                           latent_sizes=16,
                           activate_final=True,
                           with_globals=False,
                           with_senders=True,
                           normalize=True,
                           independent=False)
        model1(data)

        model2 = NodeModel(n_feat_out=16,
                           latent_sizes=16,
                           activate_final=True,
                           with_globals=False,
                           normalize=True,
                           independent=True)
        model2(data)

    def test_global_model(self):
        data = self.get_random_pyg_data()
        model1 = GlobalModel(n_feat_out=16,
                             latent_sizes=16,
                             activate_final=True,
                             normalize=True,
                             independent=False)
        model1(data)

    def test_make_mlp(self):
        data = self.get_random_pyg_data()
        model1 = GlobalModel(n_feat_out=16,
                             latent_sizes=[16, 32],
                             activate_final=True,
                             normalize=True,
                             independent=False)
        model1(data)

        model2 = GlobalModel(n_feat_out=16,
                             latent_sizes=None,
                             activate_final=True,
                             normalize=True,
                             independent=False)
        model2(data)

    def test_graph_network_model(self):
        data = self.get_random_pyg_data(n_data=4, get_first_element=False)
        edge_model_params = dict(n_feat_out=16,
                                 latent_sizes=8,
                                 activate_final=True,
                                 with_globals=False,
                                 normalize=True,
                                 independent=True)
        node_model_params = dict(n_feat_out=8,
                                 latent_sizes=2,
                                 activate_final=True,
                                 with_globals=False,
                                 with_senders=True,
                                 normalize=True,
                                 independent=False)
        global_model_params = dict(n_feat_out=5,
                                   latent_sizes=4,
                                   activate_final=True,
                                   normalize=True,
                                   independent=False)

        model1 = GraphNetwork(edge_model_params=edge_model_params,
                              node_model_params=node_model_params,
                              global_model_params=global_model_params)
        for d in data:
            o = model1(d)

        self.assertEqual(o.x.size(1), 8)
        self.assertEqual(o.u.size(1), 5)
        self.assertEqual(o.e.size(1), 16)

        model2 = GraphNetwork(edge_model_params=edge_model_params,
                              node_model_params=node_model_params)

        d = next(iter(data))
        o = model2(d)

        self.assertEqual(o.x.size(1), 8)
        self.assertEqual(o.u.size(1), 2)
        self.assertEqual(o.e.size(1), 16)

    def test_concat(self):
        data1 = self.get_random_pyg_data()
        data2 = self.get_random_pyg_data()
        model1 = NodeModel(n_feat_out=16,
                           latent_sizes=16,
                           activate_final=True,
                           with_globals=False,
                           with_senders=True,
                           normalize=True,
                           independent=False)
        model1(data1, concat_graph=data2)


if __name__ == '__main__':
    unittest.main()