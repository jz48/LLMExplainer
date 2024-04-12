import torch
import numpy as np
from utils.confidence_util.confidence_evaluating import ConfidenceEvaluator

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

_explainer = 'GNNE'


def run_llm():
    models = ['GraphGCN']
    datasets = ['ba2motif', 'mutag', 'benzene']  # 'ba2motif', 'mutag', 'benzene'

    explainers = ['PGE']  # 'PGE', 'CONFPGE'
    for d in datasets:
        for e in explainers:
            for m in models:
                evaluator = ConfidenceEvaluator(d, m, e, loss_type='ib', force_run=False,
                                                save_confidence_scores=True)
                evaluator.explainer_manager.explainer.epochs = 0
                evaluator.set_experiments()
                # evaluator.evaluating()
                # auc, auc_std, ba, ba_std, nll, nll_std = evaluator.evaluate_with_noisy()
                evaluator.show_results()

    explainers = ['LLMPGE']  # 'PGE', 'CONFPGE'
    for d in datasets:
        for e in explainers:
            for m in models:
                evaluator = ConfidenceEvaluator(d, m, e, loss_type='ib', force_run=True,
                                                save_confidence_scores=True)
                evaluator.explainer_manager.explainer.epochs = 0
                evaluator.set_experiments()
                # evaluator.evaluating()
                # auc, auc_std, ba, ba_std, nll, nll_std = evaluator.evaluate_with_noisy()
                evaluator.show_results()


if __name__ == '__main__':
    run_llm()

