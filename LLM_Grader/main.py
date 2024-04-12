import torch
import numpy as np
import random
from utils import *
from solution_manager import SolutionManager
from utils.dataset_util.dataset_loaders import DatasetLoader
from LLM_solution import LLM


class XAI_LLM(object):
    def __init__(self):
        self.dataset_name = 'ba2motif_llm'
        self.input_type = 'sparse'
        self.type = 'cls'
        self.task = 'graph'
        self.dataset_loader = DatasetLoader(self.dataset_name, self.input_type, self.type)
        self.dataset_loader.task = self.task
        self.dataset_loader.load_dataset()
        # self.dataset_loader.create_data_list()
        # self.test_set = self.build_test_set()
        self.test_set = self.build_test_set_ba2motif_grade()
        self.LLM_client = LLM(self.test_set, self.test_set)

    def build_test_set(self):
        test_sample = self.dataset_loader.graphs[0:10]
        ground_truth = self.dataset_loader.edge_ground_truth[0:10]
        labels = self.dataset_loader.labels[0:10]   # 0 means circle, 1 means house
        test_explanation = [[20, 20, 20, 21, 21, 21, 22, 22, 23, 23, 24, 24],
                            [21, 23, 24, 20, 22, 24, 21, 23, 20, 22, 20, 21]]

        data = [[test_sample[i], ground_truth[i], labels[i], test_explanation] for i in range(10)]
        return data

    def build_test_set_ba2motif_grade(self):
        random.seed(42)
        test_sample = self.dataset_loader.graphs[0:]
        ground_truth = self.dataset_loader.edge_ground_truth[0:]
        labels = self.dataset_loader.labels[0:]
        # for idx in range(len(test_sample)):
        #     test_sample[idx] = test_sample[idx].tolist()
        #     ground_truth[idx] = ground_truth[idx].tolist()
        #     test_sample[idx] = [[test_sample[idx][0][i], test_sample[idx][1][i]] for i in range(len(test_sample[idx][0])) if test_sample[idx][0][i] < test_sample[idx][1][i]]
        #     ground_truth[idx] = [[ground_truth[idx][0][i], ground_truth[idx][1][i]] for i in range(len(ground_truth[idx][0])) if ground_truth[idx][0][i] < ground_truth[idx][1][i]]

        def build_test_explanation_with_label(sample, gt, label):
            explanation = []
            grade = random.randint(0, 5)
            indices = np.where(gt == 1)
            indices = random.choices(indices[0].tolist(), k=grade)
            for idx in indices:
                explanation.append([sample[0][idx], sample[1][idx]])
            indices = np.where(gt == 0)
            indices = random.choices(indices[0].tolist(), k=5-grade)
            for idx in indices:
                explanation.append([sample[0][idx], sample[1][idx]])
            return [explanation, grade]

        test_explanations, grades = list(zip(*[build_test_explanation_with_label(test_sample[i], ground_truth[i], labels[i]) for i in range(len(test_sample))]))

        data = [[test_sample[i], ground_truth[i], labels[i], test_explanations[i], grades[i]] for i in range(len(test_sample))]
        return data

    def test(self):
        self.LLM_client.test_llm_grade(prompt_type=10)
        pass


def main():
    solution = XAI_LLM()
    solution.test()
    pass


if __name__ == "__main__":
    main()
