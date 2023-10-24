from collections import defaultdict

import scipy.stats as stats
import torch
import numpy as np

from JZJenv.Activitity import Order


# [1,1],[1,2],[1,3],[1,4]..[2,2],[2,3],[2,4],..
def HumanActions(number_humans):
    actions = []
    for cur in range(0, number_humans):
        for i in range(cur, number_humans):
            actions.append([cur, i])
    return actions
def add_lists(l1, l2):
    return [sum(x) for x in zip(l1, l2)]

def sub_lists(l1, l2):
    return [a - b for a, b in zip(l1, l2)]

def less_than(l1, l2):
    for i in range(len(l1)):
        if (l1[i] > l2[i]):
            return False
    return True

def find_index(index_list,value_list,stat='min'):
    #Function to find the index from index list whose value in value list is minimum or maximum
    if(stat=='min'):
        pos=0
        minv=value_list[0]
        for i in range(len(index_list)):
            if(value_list[i]<minv or (value_list[i]==minv and index_list[i]<index_list[pos])):
                minv=value_list[i]
                pos=i
    else: # stat='max'
        pos=0
        maxv=value_list[0]
        for i in range(len(index_list)):
            if(value_list[i]>maxv or (value_list[i]==maxv and index_list[i]<index_list[pos])):
                maxv=value_list[i]
                pos=i
    return pos
import random

#随机生成入场时间
def getST(jzjNumber):
    time = 5
    sigma =  5 / 3
    lower, upper = 0, time + sigma  # 截断在[0, μ+σ]
    X = stats.truncnorm((lower) / sigma, (upper - time) / sigma, loc=time, scale=sigma).rvs()

def generate_subgraph(graph, num_start_nodes=None):
    # 创建一个字典来存储子图的节点和紧后节点集合
    subgraph = {}
    # 创建一个集合来存储已处理的节点
    visited = set()

    if num_start_nodes is None:
        # 如果没有指定起始节点数量，则默认选择一个起始节点
        num_start_nodes = 1
    # 选择任意多个起始节点
    start_nodes = random.choices(list(graph.keys()), k=num_start_nodes)
    # 将起始节点添加到子图中并标记为已处理
    for start_node in start_nodes:
        subgraph[start_node] = graph[start_node]
        visited.add(start_node)
    # 使用深度优先搜索(Depth-First Search, DFS)生成子图
    for start_node in start_nodes:
        dfs(graph, start_node, subgraph, visited)

    return subgraph

def dfs(graph, node, subgraph, visited):
    for successor in graph[node]:
        if successor not in visited:
            # 将节点的紧后节点添加到子图中并标记为已处理
            subgraph[successor] = graph[successor]
            visited.add(successor)

            # 递归调用DFS遍历紧后节点
            dfs(graph, successor, subgraph, visited)
def truncnorm(time):
        sigma = time/3
        lower, upper = time -  sigma, time +  sigma  # 截断在[μ-2σ, μ+2σ]
        X = stats.truncnorm((lower - time) / sigma, (upper - time) / sigma, loc=time, scale=sigma).rvs()
        return X
def uniform(time):
        sigma = time/3
        lower, upper = time -  sigma, time +  sigma  # 截断在[μ-2σ, μ+2σ]
        X = random.uniform(lower, upper)
        return X



class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape):  # shape:the dimension of input data
        self.n = 0
        self.mean = np.zeros(shape)
        self.S = np.zeros(shape)
        self.std = np.sqrt(self.S)
    def update(self, x):
        x = np.array(x)
        self.n += 1
        if self.n == 1:
            self.mean = x
            self.std = x
        else:
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)
class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)
    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=False
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)
        return x
class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)
    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x
    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)

if __name__ == '__main__':
    # 测试示例
    graph = {
        'A': ['B', 'C'],
        'B': ['C', 'D'],
        'C': ['D'],
        'D': []
    }

    subgraph = generate_subgraph(graph, num_start_nodes=random.randint(2, len(graph.keys())))
    print(subgraph)