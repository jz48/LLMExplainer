from explainers.GNNExplainer import GNNExplainer
from explainers.PGExplainer import PGExplainer
from explainers.GRAD_Explainer import GRADExplainer
from explainers.GAT_Explainer import GATExplainer
from data.LLMExplainer import LLMExplainer
from explainers.LLM_random_PGExplainer import LLMRPGExplainer


class ExplainerSelector:
    def __init__(self, explainer_name, model_name, dataset_name, model_to_explain,
                 loss_type, graphs, features):
        self.explainer_name = explainer_name

        self.model_name = model_name
        if model_name in ['GraphGCN', 'GAT']:
            self.task = 'graph'
        else:
            self.task = 'node'

        self.dataset_name = dataset_name
        if dataset_name in ['bareg1', 'bareg2', 'bareg3', 'crippen', 'triangles', 'triangles_small']:
            self.model_type = 'reg'
        else:
            self.model_type = 'cls'

        self.model_to_explain = model_to_explain
        self.graphs = graphs
        self.features = features

        self.loss_type = loss_type
        self.explainer = self.select_explainer_model()

    def select_explainer_model(self):
        if self.explainer_name == 'GNNE':
            return GNNExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                self.loss_type)
        elif self.explainer_name == 'MIXGNNE':
            return MixUpGNNExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                     self.loss_type)
        elif self.explainer_name == 'NAMGNNE':
            return NAMGNNExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                   self.loss_type)
        elif self.explainer_name == 'CTLGNNE':
            return CTLGNNExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                   self.loss_type)
        elif self.explainer_name == 'PGE':
            return PGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                               self.loss_type)
        elif self.explainer_name == 'LLMRPGE':
            return LLMRPGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                  self.loss_type)
        elif self.explainer_name == 'LLM3PGE':
            return LLM3PGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                  self.loss_type)
        elif self.explainer_name == 'LLMPGE':
            return LLMExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                  self.loss_type)
        elif self.explainer_name == 'LLM1PGE':
            return LLM1PGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                  self.loss_type)
        elif self.explainer_name == 'LLM0PGE':
            return LLM0PGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                  self.loss_type)
        elif self.explainer_name == 'MIXPGE':
            return MixUpPGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                    self.loss_type)
        elif self.explainer_name == 'MIXSFTPGE':
            return MixUpSFTPGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                       self.loss_type)
        elif self.explainer_name == 'CTLPGE':
            return CTLPGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                  self.loss_type)
        elif self.explainer_name == 'CTLMIXSFTPGE':
            return CTLMixUpSFTPGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                          self.loss_type)
        elif self.explainer_name == 'CTLALTMIXPGE':
            return CTLALTMixupPGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                          self.loss_type)
        elif self.explainer_name == 'NAMPGE':
            return NAMPGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                  self.loss_type)
        elif self.explainer_name == 'GRAD':
            return GRADExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                 self.loss_type)
        elif self.explainer_name == 'GAT':
            return GATExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                self.loss_type)
        elif self.explainer_name == 'CTLPGE_no_mix':
            return CTLPGExplainer_no_mix(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                         self.loss_type)
        elif self.explainer_name == 'CTLPGE_no_cl':
            return CTLPGExplainer_no_contrastive(self.model_to_explain, self.graphs, self.features, self.task,
                                                 self.model_type, self.loss_type)
        elif self.explainer_name == 'CTLPGE_no_mse':
            return CTLPGExplainer_no_mse(self.model_to_explain, self.graphs, self.features, self.task,
                                         self.model_type, self.loss_type)
        elif self.explainer_name == 'CTLMIXSFTPGE_no_mix':
            return CTLMixUpSFTPGExplainer_no_mix(self.model_to_explain, self.graphs, self.features, self.task,
                                                 self.model_type, self.loss_type)
        elif self.explainer_name == 'CTLMIXSFTPGE_no_cl':
            return CTLMixUpSFTPGExplainer_no_cl(self.model_to_explain, self.graphs, self.features, self.task,
                                                self.model_type, self.loss_type)
        elif self.explainer_name == 'CTLMIXSFTPGE_no_mse':
            return CTLMixUpSFTPGExplainer_no_mse(self.model_to_explain, self.graphs, self.features, self.task,
                                                 self.model_type, self.loss_type)
        elif self.explainer_name == 'CIPGE':
            return CIPGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                 self.loss_type)
        elif self.explainer_name == 'CONFPGE':
            return ConfPGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                   self.loss_type)
        elif self.explainer_name == 'CONF2PGE':
            return Conf2PGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                    self.loss_type)
        elif self.explainer_name == 'CONF3PGE':
            return Conf3PGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                    self.loss_type)
        elif self.explainer_name == 'CONF4PGE':
            return Conf4PGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                    self.loss_type)
        elif self.explainer_name == 'CONF5PGE':
            return Conf5PGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                    self.loss_type)
        elif self.explainer_name == 'CONF6PGE':
            return Conf6PGExplainer(self.model_to_explain, self.graphs, self.features, self.task, self.model_type,
                                    self.loss_type)
        else:
            assert 0
