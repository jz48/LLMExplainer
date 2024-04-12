import json
import os

import torch
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
from matplotlib import pyplot as plt
from utils.training import GNNTrainer
from utils.explaining import MainExplainer
from utils.embedding_evaluating import MainEvaluator
from utils.sub_evaluating_1 import SubEvaluator_1

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

_explainer = 'GNNE'


def corrfunc(x, y, **kws):
    r, p = stats.pearsonr(x, y)
    p_stars = ''
    if p <= 0.05:
        p_stars = '*'
    if p <= 0.01:
        p_stars = '**'
    if p <= 0.001:
        p_stars = '***'
    ax = plt.gca()
    ax.annotate('r = {:.2f} '.format(r) + p_stars,
              xy=(0.05, 0.8), xycoords=ax.transAxes)


def annotate_colname(x, **kws):
    ax = plt.gca()
    ax.annotate(x.name, xy=(0.05, 0.9), xycoords=ax.transAxes,
              fontweight='bold', fontsize=20)


def cor_matrix(df, dataset):
    g = sns.PairGrid(df, palette=['red'])
    # Use normal regplot as `lowess=True` doesn't provide CIs.
    g.map_upper(sns.regplot, scatter_kws={'s':10})
    g.map_diag(sns.distplot)
    # g.map_diag(annotate_colname)
    g.map_lower(sns.kdeplot)  # , cmap='Blues_d')
    g.map_lower(corrfunc)
    # Remove axis labels, as they're in the diagonals.
    # for ax in g.axes.flatten():
    #     ax.set_ylim(-1, 1)
    #     # ax.set_xlabel('')
    g.fig.subplots_adjust(wspace=0, hspace=0)
    plt.savefig("corr_"+dataset+".png")
    return g


def evaluate_1():
    datasets = ['bareg1', 'bareg2', 'crippen', 'triangles', 'triangles_small']
    model = 'GraphGCN'
    explainer_name = 'GNNE'

    t = 0
    data_collect = []
    for dataset in datasets:
        evaluator = SubEvaluator_1(dataset, model, explainer_name)
        results = evaluator.estimate()

        results.sort(key=lambda x: x[0])
        data_collect.append([[i[0].tolist(), i[1], i[2]] for i in results])

        continue
        x = [i for i in range(len(results))]
        y = [i[0] for i in results]
        y1 = [i[1] for i in results]
        y2 = [i[2] for i in results]

        plt.figure(figsize=(10, 10))
        plt.rcParams['font.size'] = 30

        plt.plot(x, y, 'b.', label='label y')
        plt.plot(x, y1, 'r*', label='original graph $f(G)$')
        plt.plot(x, y2, 'g.', label='Explanation $f(G^*)$')

        # plt.title(dataset)
        plt.xlabel('sorted graph index')
        plt.ylabel('regression value')
        plt.legend()
        t += 1
        path = '/Users/jiaxingzhang/Desktop/Lab/XAI_GNN/XAIG_REG/data/results/figs'
        fp = os.path.join(path, 'sub_1_' + dataset + '.png')
        plt.tight_layout()
        plt.savefig(fp, transparent=True, bbox_inches='tight', pad_inches=0)

    with open('./ood_results.json', 'w') as f:
        f.write(json.dumps(data_collect))

def evaluate_2():
    datasets = ['bareg1', 'bareg2', 'crippen', 'triangles', 'triangles_small']
    # datasets = ['bareg1']
    model = 'GraphGCN'
    explainer_name = 'GNNE'

    t = 0
    for dataset in datasets:
        evaluator = SubEvaluator_1(dataset, model, explainer_name)
        results = evaluator.estimate()

        results.sort(key=lambda x: x[0])

        x = [i for i in range(len(results))]
        y = [i[0] for i in results]
        y1 = [i[1] for i in results]
        y2 = [i[2] for i in results]

        dy1 = [y1[i] - y[i] for i in range(len(x))]
        dy2 = [y2[i] - y[i] for i in range(len(x))]
        dy3 = [y2[i] - y1[i] for i in range(len(x))]
        d1 = 0
        d2 = 0
        d3 = 0
        for i in range(len(x)):
            # d1 += abs(y1[i] - y[i])
            # d2 += abs(y2[i] - y[i])
            # d3 += abs(y2[i] - y1[i])

            d1 += (y1[i] - y[i]) ** 2
            d2 += (y2[i] - y[i]) ** 2
            d3 += (y2[i] - y1[i]) ** 2

        d1 = (d1 / len(x)) ** 0.5
        d2 = (d2 / len(x)) ** 0.5
        d3 = (d3 / len(x)) ** 0.5

        print(dataset, 'd y_pred, label', d1, 'd exp label', d2, 'd exp y_pred', d3)

        data = {
            '$Y$': y,
            '$\Delta(f(G), Y)$': dy1,
            '$\Delta(f(G^*), Y)$': dy2,
            '$\Delta(f(G), f(G^*))$': dy3,
        }

        df = pd.DataFrame(data)
        df = df.iloc[1:]

        plt.rcParams.update({'font.size': 22})
        # with regression
        sns.pairplot(df, kind="reg")
        plt.savefig("correlation.png")

        plt.rcParams.update({'font.size': 15})
        # sns.set(font_scale=1.25)
        # sns.set_context("talk")
        sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.0})

        g = cor_matrix(df, dataset)

        # assert 0

        '''
        plt.figure(figsize=(10, 10))
        plt.rcParams['font.size'] = 30

        plt.plot(x, y, 'b.', label='label y')
        plt.plot(x, y1, 'r*', label='original graph f(G)')
        plt.plot(x, y2, 'g.', label='Explanation f(G^*)')

        # plt.title(dataset)
        plt.xlabel('sorted graph index')
        plt.ylabel('regression value')
        plt.legend()
        t += 1
        path = '/Users/jiaxingzhang/Desktop/Lab/XAI_GNN/XAIG_REG/data/results/figs'
        fp = os.path.join(path, 'sub_1_' + dataset + '.png')
        plt.tight_layout()
        plt.savefig(fp, transparent=True, bbox_inches='tight', pad_inches=0)
        '''


def main():
    evaluate_1()


if __name__ == '__main__':
    main()
