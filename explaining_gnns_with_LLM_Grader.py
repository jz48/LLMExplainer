import torch
import numpy as np
from LLM_Grader.LLMExplaining import LLMExplaining

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)


def explain_(dataset, model, explainer_name):
    explainer = LLMExplaining(dataset, model, explainer_name)
    auc, auc_std, rdd, rdd_std = explainer.explain()
    print(dataset, model, explainer_name, 'auc: ', auc, 'auc_std: ', auc_std, 'rdd: ', rdd, 'rdd_std: ', rdd_std)


def runs():
    datasets = ['mutag']  # 'mutag', 'flca', 'alca', 'bareg2', 'bareg1', 'ba2motif'

    explainers = ['LLM2PGE']   # 'CTLGNNE', 'MIXGNNE', 'NAMGNNE', 'NAMPGE', 'MIXPGE'
    models = ['GraphGCN']
    loss_type = 'ib'
    for d in datasets:
        for e in explainers:
            for m in models:
                explainer = LLMExplaining(d, m, e, wandb_log=False, loss_type=loss_type, force_run=False,
                                          save_explaining_results=False)
                explainer.explainer_manager.explainer.epochs = 100
                explainer.explain()
                explainer.show_results()


if __name__ == '__main__':
    runs()
