import json
import math
import os
import numpy
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from utils.dataset_util.dataset_loaders import DatasetLoader
from utils.dataset_util.data_utils import to_torch_graph
from gnns.model_selector import ModelSelector
from explainers.explainer_selector import ExplainerSelector
from utils.wandb_logger import WandbLogger


class LLMExplaining(object):
    def __init__(self, dataset_name, model_name, explainer_name, wandb_log=False, loss_type='ib',
                 sample_bias=0.0, thres_snip=5, thres_min=-1
                 , temps=None, seeds=None, save_explaining_results=True, force_run=False):

        if seeds is None:
            self.seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        if temps is None:
            self.temps = [5.0, 1.0]

        self.dataset_name = dataset_name
        self.model_name = model_name
        self.explainer_name = explainer_name
        self.logging = wandb_log
        self.loss_type = loss_type

        self.seed = seeds
        self.force_run = force_run
        self.save_explaining_results = save_explaining_results

        if model_name == 'RegGCN':
            self.input_type = 'dense'
        else:
            self.input_type = 'sparse'

        if dataset_name in ['bareg1', 'bareg2', 'crippen', 'bareg3', 'triangles', 'triangles_small']:
            self.type = 'reg'
        else:
            self.type = 'cls'

        if dataset_name in ['syn1', 'syn2', 'syn3', 'syn4', ]:
            self.task = 'node'
        else:
            self.task = 'graph'

        self.dataset_loader = DatasetLoader(self.dataset_name, self.input_type, self.type)
        self.dataset_loader.task = self.task
        self.dataset_loader.load_dataset()
        self.dataset_loader.create_data_list()

        self.graphs, self.features = self.dataset_loader.graphs, self.dataset_loader.features
        self.labels, self.test_mask = self.dataset_loader.labels, self.dataset_loader.test_mask
        self.ground_truth = self.dataset_loader.edge_ground_truth

        self.model_manager = ModelSelector(model_name, dataset_name, load_pretrain=True)
        self.model_manager.model.type = self.type
        self.explainer_manager = ExplainerSelector(explainer_name, model_name, dataset_name, self.model_manager.model,
                                                   self.loss_type, self.graphs, self.features)
        project_name = explainer_name + '_' + dataset_name + '_' + model_name
        self.explainer_manager.explainer.wandb_logger = WandbLogger(project_name, self.explainer_manager.explainer.config, wandb_log)
        self.explainer_manager.explainer.total_step = 0

    def grade_explanation(self, explanation, ground_truth):
        pass

    def explain(self):
        if self.dataset_name in ['crippen', 'flca', 'benzene', 'alca']:
            self.explainer_manager.explainer.features = [torch.tensor(i) for i in self.features]
        else:
            self.explainer_manager.explainer.features = torch.tensor(self.features)

        self.explainer_manager.explainer.labels = torch.tensor(self.labels)
        self.explainer_manager.explainer.graphs = to_torch_graph(self.graphs, self.explainer_manager.task)
        self.explainer_manager.explainer.ground_truth = [torch.tensor(i) for i in self.ground_truth]

        # Get ground_truth for every node
        indices = np.argwhere(self.dataset_loader.val_mask).squeeze()
        if self.logging:
            self.seeds = [0, 1, 2]
            indices = indices[:10]
        # Perform the evaluation 10 times
        auc_scores = []
        rdd_scores = []

        folder_path = os.path.join(self.dataset_loader.data_root_path, 'results/confidence_evaluation/logs/',
                                   self.explainer_name, self.dataset_name, self.model_name,
                                   self.explainer_manager.explainer.serialize_configuration())

        save_score_path = os.path.join(self.dataset_loader.data_root_path, 'results/scores/',
                                       self.explainer_name, self.dataset_name, self.model_name,
                                       self.explainer_manager.explainer.serialize_configuration())

        if os.path.exists(save_score_path) and not self.force_run:
            print('already finished!')
            return self.show_results()

        if not os.path.exists(save_score_path):
            os.makedirs(save_score_path)

        scores = []

        for turn_, s in enumerate(self.seeds):
            print(f"Run {turn_} with seed {s}")
            # Set all seeds needed for reproducibility
            torch.manual_seed(s)
            torch.cuda.manual_seed(s)
            np.random.seed(s)

            self.explainer_manager.explainer.prepare(indices)
            self.explainer_manager.explainer.dataset_name = self.dataset_name
            self.explainer_manager.explainer.train_res_dict = {}
            score_dict = {}
            record_each_epoch = True
            if record_each_epoch:
                for e in tqdm(range(0, self.explainer_manager.explainer.epochs)):
                    loss = self.explainer_manager.explainer.train_1_epoch(e, indices)
                    explanations = []
                    for idx in tqdm(indices, disable=(not self.logging)):
                        graph, expl = self.explainer_manager.explainer.explain(idx)
                        explanations.append((idx, graph, expl))
                    self.explainer_manager.explainer.train_res_dict[e] = {'e': e, 'explanations': explanations,
                                                                          'train_loss': loss}
                    y_true, y_pred = self.prepare_evaluate(explanations)
                    auc_score = self.evaluate_auc(y_true, y_pred)
                    llm_score = 0
                    for i in range(0, len(explanations)):
                        idx, graph, expl = explanations[i]
                        y_true = self.explainer_manager.explainer.ground_truth[idx]
                        llm_score += self.explainer_manager.explainer.grade_explanation(expl, y_true)
                    llm_score = llm_score / len(explanations)
                    score_dict[e] = {'auc': auc_score, 'llm': llm_score, 'train_loss': loss}
                    # print("auc score:", auc_score, "llm score: ", llm_score, "train loss: ", loss)
            else:
                self.explainer_manager.explainer.train(indices)

            explanations = []
            for idx in tqdm(indices, disable=(not self.logging)):
                graph, expl = self.explainer_manager.explainer.explain(idx)
                explanations.append((idx, graph, expl))

            self.explainer_manager.explainer.train_res_dict[self.explainer_manager.explainer.epochs-1]['explanations'] = explanations

            if record_each_epoch:
                file_name = str(s) + '_exp_results_by_epoch.json'
                save_path = folder_path
                for _, key in enumerate(self.explainer_manager.explainer.train_res_dict):
                    self.explainer_manager.explainer.train_res_dict[key]['explanations'] = \
                        [[int(key), i[1].detach().numpy().tolist(), i[2].detach().numpy().tolist()] for i in
                         self.explainer_manager.explainer.train_res_dict[key]['explanations']]
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                path = os.path.join(save_path, file_name)
                with open(path, 'w') as f:
                    f.write(json.dumps(self.explainer_manager.explainer.train_res_dict))

                score_dict_file_name = str(s) + '_score_dict_by_epoch.json'
                path = os.path.join(save_path, score_dict_file_name)
                with open(path, 'w') as f:
                    f.write(json.dumps(score_dict))

            if False:
                file_name = str(s) + '_exp_results_training.json'
                save_path = folder_path
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                path = os.path.join(save_path, file_name)
                with open(path, 'w') as f:
                    f.write(json.dumps(self.explainer_manager.explainer.train_res_dict))

            y_true, y_pred = self.prepare_evaluate(explanations)
            auc_score = self.evaluate_auc(y_true, y_pred)
            rdd_score = self.evaluate_rdd(y_true, y_pred)
            print("auc score:", auc_score, "rdd score: ", rdd_score)

            scores.append([s, auc_score, rdd_score])

        with open(os.path.join(save_score_path, 'scores.json'), 'w') as f:
            f.write(json.dumps(scores))

        return self.show_results()

    def prepare_evaluate(self, explanations):
        y_true = []
        y_pred = []
        for idx, graph_, expl_ in explanations:
            graph = graph_.detach().numpy()
            expl = expl_.detach().numpy()
            ground_truth = self.dataset_loader.edge_ground_truth[idx]

            for edge_idx in range(0, expl.shape[0]):  # Consider every edge in the ground truth
                edge_ = graph.T[edge_idx]
                if edge_[0] == edge_[1]:  # We dont consider self loops for our evaluation (Needed for Ba2Motif)
                    continue
                # Retrieve predictions and ground truth
                y_pred.append(expl[edge_idx])
                y_true.append(ground_truth[edge_idx])
        return y_true, y_pred

    def evaluate_auc(self, y_true, y_pred):
        return roc_auc_score(y_true, y_pred)

    def evaluate_rdd(self, ground_truth, explanation):
        """
        :param: explanation: edge weight vector
        :param: ground_truth: binary/float edge ground truth
        evaluate the reversed distribution distance from explanation to ground truth by f(x_{N})= 1 - root(x_{i}^{2})/N
        :return: float
        """
        assert len(explanation) == len(ground_truth)
        sum_d = 0.0
        for i in range(len(explanation)):
            sum_d += (explanation[i] - ground_truth[i]) ** 2
        return 1 - (sum_d / len(explanation)) ** 0.5

    def evaluation_auc_(self, explanations):
        """Determines based on the task which auc evaluation method should be called to determine the AUC score

        :param task: str either "node" or "graph".
        :param explanations: predicted labels.
        :param ground_truth: ground truth labels.
        :param indices: Which indices to evaluate. We ignore all others.
        :returns: area under curve score.
        """
        if self.explainer_manager.explainer.task == 'graph':
            return self.evaluation_auc_graph(explanations)
        elif self.explainer_manager.explainer.task == 'node':
            return self.evaluation_auc_node(explanations)

    def evaluation_auc_graph(self, explanations):
        """Evaluate the auc score given explaination and ground truth labels.

        :param explanations: predicted labels.
        :param ground_truth: ground truth labels.
        :param indices: Which indices to evaluate. We ignore all others.
        :returns: area under curve score.
        """
        ys = []
        predictions = []

        for i in explanations:
            idx = i[0]
            graph = i[1].detach().numpy()
            expl = i[2].detach().numpy()
            ground_truth = self.dataset_loader.edge_ground_truth[idx]

            for edge_idx in range(0, expl.shape[0]):  # Consider every edge in the ground truth
                edge_ = graph.T[edge_idx]
                if edge_[0] == edge_[1]:  # We dont consider self loops for our evaluation (Needed for Ba2Motif)
                    continue

                # Retrieve predictions and ground truth
                predictions.append(expl[edge_idx])
                ys.append(ground_truth[edge_idx])
            a = 0

        score = roc_auc_score(ys, predictions)
        return score

    def show_results(self):
        save_score_path = os.path.join(self.dataset_loader.data_root_path, 'results/scores/',
                                       self.explainer_name, self.dataset_name, self.model_name,
                                       self.explainer_manager.explainer.serialize_configuration())
        with open(os.path.join(save_score_path, 'scores.json'), 'r') as f:
            scores = json.loads(f.read())
        auc_scores = [i[1] for i in scores]
        rdd_scores = [i[2] for i in scores]
        auc = np.mean(auc_scores)
        auc_std = np.std(auc_scores)
        rdd = np.mean(rdd_scores)
        rdd_std = np.std(rdd_scores)
        print(self.explainer_name, self.dataset_name, self.model_name)
        print("Auc score: %.4f (%.4f)" % (auc, auc_std))
        print("RDD score: %.4f (%.4f)" % (rdd, rdd_std))

        return {'auc': auc, 'auc_std': auc_std, 'rdd': rdd, 'rdd_std': rdd_std}
