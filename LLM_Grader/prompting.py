
class Prompter(object):
    prompt_type = -1
    prompt_count = 0

    def __init__(self):
        self.prompt_type = 1
        self.prompt_count = 0
        pass

    def prompt(self, sample, prompt_type=1):
        self.prompt_type = prompt_type
        if self.prompt_type == 0:
            return self.build_graph_query_0(sample)
        if self.prompt_type == 1:
            return self.build_graph_query_1(sample)
        elif self.prompt_type == 2:
            return self.build_graph_query_2(sample)
        elif self.prompt_type == 3:
            return self.build_graph_query_3(sample)
        elif self.prompt_type == 4:
            return self.build_graph_query_4(sample)
        elif self.prompt_type == 5:
            return self.build_graph_query_5(sample)
        elif self.prompt_type == 6:
            return self.build_graph_query_6(sample)
        elif self.prompt_type == 8:
            return self.build_graph_query_8(sample)
        elif self.prompt_type == 10:
            return self.build_graph_query_10(sample)

    def build_graph_query_0(self, sample):
        var1 = sample[0].tolist()
        var2 = [[var1[0][i] for i in range(len(sample[1])) if sample[1][i] == 1], [var1[1][i] for i in range(len(sample[1])) if sample[1][i] == 1]]
        var2 = '[' + ' '.join([str(i) for i in var2]) + ']'  # explanation
        var1 = '[' + ' '.join([str(i) for i in var1]) + ']'  # original graph
        prompt = "I will give you a graph 'G' and a sub-graph 'G^*'. " \
                 "Your task is check whether or not a given sub-graph 'G*' exist in a given graph 'G' \n" \
                 "The graph is represented by an edge index list, which contains the source nodes and targets nodes in the graph." \
                 "For example: \n '''[[1,2,3], [2, 3, 1] means a graph with three edges, these is an edge from 1 to node 2;" \
                 "an edge from node 2 to node 3 and an edge from node 3 to node 1. " \
                 "This graph is also a triangle or circle structure. ''' \n" \
                 "Now, I'm giving you the graph 'G' and sub-graph 'G*', " \
                 "Please tell me yes or no the sub-graph 'G*' exists in the graph 'G'. \n" \
                 "The graph 'G' is: ''' " + var1 + "''' \n" \
                 "The sub-graph 'G*' is: '''" + var2 + "''' \n" \

        print(prompt)
        self.prompt_count += 1
        return prompt

    def build_graph_query_2(self, sample):
        var1 = sample[0].tolist()
        var2 = [[var1[0][i] for i in range(len(sample[1])) if sample[1][i] == 1], [var1[1][i] for i in range(len(sample[1])) if sample[1][i] == 1]]
        var2 = '[' + ' '.join([str(i) for i in var2]) + ']'  # explanation
        var1 = '[' + ' '.join([str(i) for i in var1]) + ']'  # original graph
        prompt = "I will give you a graph 'G' and a sub-graph 'G^*'. " \
                 "Your task is check whether or not a given sub-graph 'G*' exist in a given graph 'G' \n" \
                 "The graph is represented by an edge index list, which contains the source nodes and targets nodes in the graph." \
                 "For example: \n '''[[1,2,3], [2, 3, 1] means a graph with three edges, these is an edge from 1 to node 2;" \
                 "an edge from node 2 to node 3 and an edge from node 3 to node 1. " \
                 "This graph is also a triangle or circle structure. ''' \n" \
                 "Now, I'm giving you the graph 'G' and sub-graph 'G*', " \
                 "Please tell me yes or no the sub-graph 'G*' exists in the graph 'G'. \n" \
                 "The graph 'G' is: ''' " + var1 + "''' \n" \
                 "The sub-graph 'G*' is: '''" + var2 + "''' \n" \
                 "REMEMBER IT: Keep your answer short!"
        print(prompt)
        self.prompt_count += 1
        return prompt

    def build_graph_query_4(self, sample):
        var1 = sample[0].tolist()
        var2 = sample[1]

        var1 = [[aa, bb] for aa, bb in zip(var1[0], var1[1])]
        var2 = [[aa, bb] for aa, bb in zip(var2[0], var2[1])]

        var2 = '[' + ' '.join([str(i) for i in var2]) + ']'  # explanation
        var1 = '[' + ' '.join([str(i) for i in var1]) + ']'  # original graph

        prompt = "I will give you a graph 'G' and a sub-graph 'G^*'. " \
                 "Your task is check whether or not a given sub-graph 'G*' exist in a given graph 'G' \n" \
                 "The graph is represented by an edge index list, which contains the source nodes and targets nodes in the graph." \
                 "For example: \n " \
                 "'''[[1,2], [2, 3], [3, 1]] means a graph with three edges, these is an edge from 1 to node 2;" \
                 "an edge from node 2 to node 3 and an edge from node 3 to node 1. " \
                 "This graph is also a triangle or circle structure. ''' \n" \
                 "'''[[1,2], [2, 3], [3, 4], [4, 5], [5, 1]] means a graph with five edges, these is an edge from 1 to node 2;" \
                 "an edge from node 2 to node 3 and an edge from node 3 to node 1, etc... " \
                 "This graph is also a five nodes circle structure. ''' \n"\
                 "'''[[1,2], [2, 3], [1, 3], [3, 4], [4, 5], [5, 1]] means a graph with six edges, these is an edge from 1 to node 2;" \
                 "an edge from node 2 to node 3 and an edge from node 3 to node 1, etc... " \
                 "This graph is also a house structure. ''' \n" \
                 "A graph with circle structure sub-graph is mark as negative sample and a graph with house structure sub-graph is positive structure. \n" \
                 "The node id of the structure may change, but the relation of the edges wouldn't change. \n" \
                 "Now, I'm giving you the graph 'G' and sub-graph 'G*', \n" \
                 "The graph 'G' is: ''' " + var1 + "''' \n" \
                 "The sub-graph 'G*' is: '''" + var2 + "''' \n" \
                 "1. Please tell me which structure the sub-graph 'G*' is. \n" \
                 "2. Please tell me positive or negative the graph is. \n" \
                 "3. Please tell me yes or no the sub-graph 'G*' exists in the graph 'G'. \n"
                 # "REMEMBER IT: Keep your answer short!"
        print(prompt)
        self.prompt_count += 1
        return prompt

    def build_graph_query_6(self, sample):
        # test negative sample
        var1 = sample[0].tolist()
        var2 = sample[1]

        var1 = [[aa, bb] for aa, bb in zip(var1[0], var1[1])]
        var2 = [[aa, bb] for aa, bb in zip(var2[0], var2[1])]

        var2 = '[' + ' '.join([str(i) for i in var2]) + ']'  # explanation
        var1 = '[' + ' '.join([str(i) for i in var1]) + ']'  # original graph
        prompt = "I will give you a graph 'G' and a sub-graph 'G^*'. " \
                 "Your task is check whether or not a given sub-graph 'G*' exist in a given graph 'G' \n" \
                 "The graph is represented by an edge index list, which contains the source nodes and targets nodes in the graph." \
                 "For example: \n '''[[1,2], [2,3], [3,1] means a graph with three edges, these is an edge from 1 to node 2;" \
                 "an edge from node 2 to node 3 and an edge from node 3 to node 1. " \
                 "This graph is also a triangle or circle structure. ''' \n" \
                 "Now, I'm giving you the graph 'G' and sub-graph 'G*', " \
                 "Please tell me yes or no the sub-graph 'G*' exists in the graph 'G'. \n" \
                 "The graph 'G' is: ''' " + var1 + "''' \n" \
                 "The sub-graph 'G*' is: '''" + var2 + "''' \n" # \
                 # "You may think step by step, but I don't need the intermediate steps. Instead, I only want to have the final answer. REMEMBER IT: Keep your answer short!"
        #  print(prompt)

        #  emphsize the structure in the graph prompt
        #  emphsize llm not self-correcting the wrong answer
        self.prompt_count += 1
        return prompt

    def build_graph_query_8(self, sample):
        var1 = sample[0].tolist()
        var2 = sample[1]

        var1 = [[aa, bb] for aa, bb in zip(var1[0], var1[1]) if aa < bb]
        var2 = [[aa, bb] for aa, bb in zip(var2[0], var2[1]) if aa < bb]

        var2 = '[' + ' '.join([str(i) for i in var2]) + ']'  # explanation
        var1 = '[' + ' '.join([str(i) for i in var1]) + ']'  # original graph

        prompt = "I will give you a graph 'G' and a sub-graph 'G^*'. " \
                 "Your task is check whether or not a given sub-graph 'G*' exist in a given graph 'G' \n" \
                 "The graph is represented by an edge index list, which contains the source nodes and targets nodes in the graph." \
                 "For example: \n " \
                 "'''[[1,2], [2, 3], [3, 1]] means a graph with three edges, these is an edge from 1 to node 2;" \
                 "an edge from node 2 to node 3 and an edge from node 3 to node 1. " \
                 "This graph is also a triangle or circle structure. ''' \n" \
                 "'''[[1,2], [2, 3], [3, 4], [4, 5], [5, 1]] means a graph with five edges, these is an edge from 1 to node 2;" \
                 "an edge from node 2 to node 3 and an edge from node 3 to node 1, etc... " \
                 "This graph is also a five nodes circle structure. ''' \n"\
                 "'''[[1,2], [2, 3], [1, 3], [3, 4], [4, 5], [5, 1]] means a graph with six edges, these is an edge from 1 to node 2;" \
                 "an edge from node 2 to node 3 and an edge from node 3 to node 1, etc... " \
                 "This graph is also a house structure. ''' \n" \
                 "A graph with circle structure sub-graph is mark as negative sample and a graph with house structure sub-graph is positive structure. \n" \
                 "The node id of the structure may change, but the relation of the edges wouldn't change. \n" \
                 "Now, I'm giving you the graph 'G' and sub-graph 'G*', \n" \
                 "The graph 'G' is: ''' " + var1 + "''' \n" \
                 "The sub-graph 'G*' is: '''" + var2 + "''' \n" \
                 "1. Please tell me which structure the sub-graph 'G*' is. \n" \
                 "2. Please tell me positive or negative the graph is. \n" \
                 "3. Please tell me yes or no the sub-graph 'G*' exists in the graph 'G'. \n"
                 # "REMEMBER IT: Keep your answer short!"
        print(prompt)
        self.prompt_count += 1
        return prompt

    def build_graph_query_10(self, sample):
        var1 = sample[0].tolist()
        var2 = sample[1]
        shots_prompt = sample[2]
        var1 = [[aa, bb] for aa, bb in zip(var1[0], var1[1]) if aa < bb]
        # var2 = [[aa, bb] for aa, bb in zip(var2[0], var2[1]) if aa < bb]

        var2 = '[' + ' '.join([str(i) for i in var2]) + ']'  # explanation
        var1 = '[' + ' '.join([str(i) for i in var1]) + ']'  # original graph

        prompt = "I will give you a graph 'G' and a sub-graph 'Ge'. The ground truth explanation sub-graph 'Ge' of the original graph 'G'" \
                 "is a house motif or a circle motif. \n" \
                 "Your task is graded the given sub-graph 'Gs' for given original graph 'G' \n" \
                 "The grade scores are [0, 1, 2, 3, 4, 5], higher the score is, better the sub-graph 'Gs' close to the " \
                 "ground-truth explanation sub-graph 'Ge'\n" \
                 "The graph is represented by an edge list, which contains the source nodes and targets nodes in the graph.\n" \
                 "Here are some examples: \n" \
                 + shots_prompt + \
                 "Now, I'm giving you the graph 'G' and sub-graph 'Gs', \n" \
                 "The graph 'G' is: ''' " + var1 + "''' \n" \
                 "The sub-graph 'Gs' is: '''" + var2 + "''' \n" \
                 "Please grade the sub-graph 'Gs' is. \n" \
                 "REMEMBER IT: Keep your answer short!"
        # print(prompt)
        self.prompt_count += 1
        return prompt