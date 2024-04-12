import openai
import os
import pandas as pd
import json
import time
from datetime import datetime
from utils import *
import numpy as np
# from prompting import Prompter
from prompting_Jiayi import Prompter2

openai.api_key = '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'
openai.api_key = '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'
openai.api_key= "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"  # Jiayi's key

prompter = Prompter2()


def build_multiple_func_in_one_prompt(funcs):
    prompt = "Please act as a academic calculator. Your task is calculate the derivative of the following functions: \n"
    for i in range(len(funcs)):
        prompt += "My function " + str(i+1) + " is: '''" + funcs[i] + "''' \n"
    prompt += "Please return the derivative in the simplified form of polynomial as output."
    print(prompt)
    return prompt


def process_response(res_str):
    # todo
    return res_str


def process_multi_response(funcs, res_str):
    results = []
    # todo
    return results


class LLM(object):
    def __init__(self, train_set, test_set):
        self.train_set = train_set
        self.test_set = test_set
        self.results = []
        self.batch_size = 50
        self.cool_time = 0

        self.load_log = 1
        self.load_log_path = '../data/12_04_2023_17:40:06LLM_log_gpt35.json'
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H:%M:%S")
        print("date and time:", date_time)
        self.LLM_log_path = './'+date_time+'LLM_log_gpt35.json'

        if not os.path.exists(self.LLM_log_path):
            with open(self.LLM_log_path, 'w') as f:
                f.write('{}')
        with open(self.LLM_log_path, 'r') as f:
            self.log_dict = json.loads(f.readline())

    def save_log(self):
        with open(self.LLM_log_path, 'w') as f:
            f.write(json.dumps(self.log_dict))

    def get_completion(self, prompt, model="gpt-3.5-turbo"):
        messages = [{"role": "user", "content": prompt}]
        time.sleep(self.cool_time)
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message["content"]

    def predict(self, test_sample):
        '''
        test sample: a graph G with explanation sub-graph G^* and ground-truth g(G^*), [G, G^*, g(G^*)]
        graph: edge index [sources, targets], eg: [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 1]]
        return: a score to grade the given explanation
        '''
        graph = test_sample[0]
        explanation = test_sample[1]
        ground_truth = test_sample[2]

        if str(test_sample[0]) in self.log_dict.keys():
            response = self.log_dict[str(test_sample[0])]
        else:
            test_func, test_variable = extract_variable(test_sample[0])
            context = prompting(test_func, test_variable)
            response = self.get_completion(context)
            self.log_dict[str(test_sample[0])] = response
            print(len(self.log_dict))
            if len(self.log_dict) % 10 == 0:
                print('saving... ')
                self.save_log()
            response = process_response(response)
        self.results.append(response)
        return response

    def predict_multi_funcs(self, test_samples):
        response = []
        # todo
        return response

    def run(self):
        true_derivatives = [i[1] for i in self.test_set]
        if self.batch_size == 1:
            predicted_derivatives = [self.predict(f) for f in self.test_set]
        else:
            predicted_derivatives = []
            test_set = self.test_set.copy()
            while True:
                if len(test_set) >= self.batch_size:
                    batch_samples = test_set[:self.batch_size]
                    test_set = test_set[self.batch_size:]
                    predicted_derivatives.extend(self.predict_multi_funcs(batch_samples))
                else:
                    predicted_derivatives.extend(self.predict_multi_funcs(test_set))
                    break
        # scores = [score(td, pd) for td, pd in zip(true_derivatives, predicted_derivatives)]
        # print(np.mean(scores))
        pass

    def test(self, prompt_type=1):
        data = self.test_set
        results = []
        for i in range(len(data)):
            test_sample = data[i]
            if test_sample[2] == 0:
                print('circle motif')
            elif test_sample[2] == 1:
                print('house motif')
            prompt = prompter.prompt([test_sample[0], test_sample[3]], prompt_type=prompt_type)
            response = self.get_completion(prompt)
            results.append([str(test_sample[2]), response])
            print(response)
            with open(self.LLM_log_path, 'w') as f:
                f.write(json.dumps(results))
        pass

    def test_llm_grade(self, prompt_type=1):
        data = self.test_set
        results = []
        shots = []
        if self.load_log:
            with open(self.load_log_path, 'r') as f:
                log_data = json.loads(f.readline())
            y_true = []
            y_pred = []
        for i in range(6):
            for j in data:
                if j[4] == i:
                    shots.append([i, j[3]])
                    break
        shots_prompt = 'Here are some examples of the explanation sub-graphs and related grades: \n'
        for i in range(len(shots)):
            shots_prompt += 'Example ' + str(i+1) + ': ' + str(shots[i][1]) + ', grade: ' + str(shots[i][0]) + '\n'
        shots_prompt += 'Please grade the following explanation sub-graph: \n'

        for i in range(len(data)):
            print('sample id: ', i)
            test_sample = data[i]
            if test_sample[2] == 0:
                print('circle motif')
            elif test_sample[2] == 1:
                print('house motif')
            print('grade: ', test_sample[4])

            if self.load_log:
                response = log_data[i][1]
                print(response)
                score_i = response.split(' ')[-1].split('.')[0]
                print('score: ', score_i)
                try:
                    y_pred.append(int(score_i))
                    y_true.append(test_sample[4])
                except:
                    pass
            else:
                prompt = prompter.prompt([test_sample[0], test_sample[3], shots_prompt], prompt_type=prompt_type)
                response = self.get_completion(prompt)
                print(response)
                results.append([str(test_sample[2]), response])
                with open(self.LLM_log_path, 'w') as f:
                    f.write(json.dumps(results))

        if self.load_log:
            print('MAE: ', np.mean(np.abs(np.array(y_true)-np.array(y_pred))))
            print('RMSE: ', np.sqrt(np.mean(np.square(np.array(y_true)-np.array(y_pred)))))
            print('ACC: ', np.mean(np.array(y_true)==np.array(y_pred)))
        pass

    def test_multi(self):
        pred_derivatives = self.log_dict
        responses = self.log_dict['responses']
        data = []
        gt_dict = {key: value for key, value in self.test_set}

        for funcs, responses in responses:
            if len(funcs) == 0:
                continue
            res = process_multi_response(funcs, responses)
            for i in res:
                self.log_dict[str(i[0])] = str(i[1])
                data.append([i[0], gt_dict[i[0]], i[1]])

        scores = [score(td, pd) for [_, td, pd] in data]
        print(np.mean(scores))
        pass
