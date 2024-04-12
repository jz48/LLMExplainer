import json
import math
import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data, DataLoader
from utils.dataset_util.dataset_loaders import DatasetLoader
from utils.dataset_util.data_utils import to_torch_graph
from gnns.model_selector import ModelSelector
from explainers.explainer_selector import ExplainerSelector
from utils.embedding_evaluating import MainEvaluator


class SubEvaluator_1(MainEvaluator):
    def __init__(self, dataset_name, model_name, explainer_name,
                 sample_bias=0.0, thres_snip=5, thres_min=-1, temps=None, seeds=None, ):
        super().__init__(dataset_name, model_name, explainer_name,
                         sample_bias=0.0, thres_snip=5, thres_min=-1, temps=None, seeds=None, )

    def estimate(self):
        if self.dataset_name in ['crippen']:
            self.explainer_manager.explainer.features = [torch.tensor(i) for i in self.features]
        else:
            self.explainer_manager.explainer.features = torch.tensor(self.features)
        self.explainer_manager.explainer.labels = torch.tensor(self.labels)
        self.explainer_manager.explainer.graphs = to_torch_graph(self.graphs, self.explainer_manager.task)
        sub_graphs = [torch.tensor(i) for i in self.sub_graphs]  # torch.tensor(self.sub_graphs)

        indices = np.argwhere(self.dataset_loader.val_mask).squeeze()
        results = []
        for idx in tqdm(indices):
            index = int(idx)
            feats = self.explainer_manager.explainer.features[index]
            graph = self.explainer_manager.explainer.graphs[index]
            sub_graph = sub_graphs[index]

            if self.dataset_name in ['triangles', 'triangles_small']:
                y = self.labels[index]
            else:
                y = self.labels[index][0]
            y_pred = self.explainer_manager.explainer.model_to_explain(feats, graph).detach().numpy().tolist()[0][0]
            exp_pred = self.explainer_manager.explainer.model_to_explain(feats, sub_graph).detach().numpy().tolist()[0][0]
            results.append([y, y_pred, exp_pred])
        return results


def main():
    pass


if __name__ == '__main__':
    main()
