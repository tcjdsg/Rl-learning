import random
from collections import defaultdict

import numpy as np
import torch

from JZJenv.Activitity import Order
from JZJenv.FixedMess import FixedMes
def getData(n_jzj, n_order, n_human, seed=None):
    if seed != None:
        np.random.seed(seed)
    indexHuman = [0]
    for num in n_human:
        indexHuman.append(indexHuman[-1] + num)

    num_hum = [random.randint(-20, 100) for _ in range(n_order * n_jzj)]
    exist = [1 for _ in range(n_jzj * n_order)]
    time = [0.0 for _ in range(n_jzj * n_order)]

    for j in range(n_jzj * n_order):
        if num_hum[j] <= 0:
            exist[j] = 0
            continue
        t = getTime(j % n_order)
        time[j] = t
    preActDict = defaultdict(lambda: [])
    sucActDict = defaultdict(lambda: [])
    index = 0
    # 构建任务网络
    for i in range(n_jzj):
        for j in range(n_order):
            SUCOrder = [num + n_order * i for num in FixedMes.SUCOrder[j]]
            PREOrder = [num + n_order * i for num in FixedMes.PREOrder[j]]
            sucActDict[index] = SUCOrder
            preActDict[index] = PREOrder
            index+=1
    n  = n_jzj * n_order
    adjacency_matrix = [[0 for _ in range(n)] for _ in range(n)]

    for node, successors in sucActDict.items():
        for successor in successors:
                adjacency_matrix[node][successor] = 1
    adj = torch.tensor(adjacency_matrix)
    sucActDict, preActDict = build_adjacency_matrix(sucActDict, exist, n_jzj, n_order)
    return adj, sucActDict, preActDict, time, exist
def build_adjacency_matrix(suc, exist, n_jzj, n_orders):
    """
    根据紧后节点集合和节点存在情况构建新的邻接矩阵。
    参数：
    successors：每个节点的紧后节点集合，是一个二维列表，successors[i] 表示第 i 个节点的紧后节点集合。
    exists：节点是否存在的一维数组，长度为节点个数，0 表示该节点不存在。

    返回值：
    新的邻接矩阵表示的有向图。
    """
    n = len(exist)

    adjacency_matrix = [[0 for _ in range(n)] for _ in range(n)]  # 初始化邻接矩阵

    for node, successors in suc.items():
        for successor in successors:
                adjacency_matrix[node][successor] = 1

    adj = torch.tensor(adjacency_matrix)
    # 更新紧前节点的紧后节点集合
    for i in range(n_jzj * n_orders):
        if exist[i]==0:
            for j in range(n_jzj * n_orders):
                if adj[j][i] == 1:
                    adj[j][i] = 0
                    adj[j] += adj[i]
                    for num in range(len(adj[j])):
                        if adj[j][num] > 1:
                            adj[j][num] = 1
            adj[i] = torch.tensor([0 for _ in range(n_jzj * n_orders)])

    sucActDict = defaultdict(lambda: [])

    preActDict = defaultdict(lambda: [])
    for i in range(n_jzj * n_orders):
        sucActDict[i] = []
        preActDict[i] = []
        for j in range(n_jzj * n_orders):
            if adj[i][j] == 1 and i != j:
                sucActDict[i].append(j)
            if adj[j][i] == 1 and i != j:
                preActDict[i].append(j)
    return sucActDict, preActDict


def getTime(i):
    # # 定义每种任务的时间分布
    # if i == 0:
    #     return 0
    # elif i == 1:
    #     return truncnorm(FixedMes.OrderTime[1])
    # elif i == 2:
    #     return truncnorm(FixedMes.OrderTime[2])
    # elif i == 3:
    #     return truncnorm(FixedMes.OrderTime[3])
    # elif i == 4:
    #     return truncnorm(FixedMes.OrderTime[4])
    # elif i == 5:
    #     return truncnorm(FixedMes.OrderTime[5])
    # elif i == 6:
    #     return truncnorm(FixedMes.OrderTime[6])
    # elif i == 7:
    #     return truncnorm(FixedMes.OrderTime[7])
    #
    # elif i == 8:
    #     return truncnorm(FixedMes.OrderTime[8])
    # elif i == 9:
    #     return truncnorm(FixedMes.OrderTime[9])
    #
    # elif i == 10:
    #     return truncnorm(FixedMes.OrderTime[10])
    # elif i == 11:
    #     return truncnorm(FixedMes.OrderTime[11])
    #
    # elif i == 12:
    #     return truncnorm(FixedMes.OrderTime[12])
    # elif i == 13:
    #     return truncnorm(FixedMes.OrderTime[13])
    # elif i == 14:
    #     return truncnorm(FixedMes.OrderTime[14])
    #
    # elif i == 15:
    #     return truncnorm(FixedMes.OrderTime[15])
    # elif i == 16:
    #     return truncnorm(FixedMes.OrderTime[16])
    # elif i == 17:
    #     return truncnorm(FixedMes.OrderTime[17])
    #
    # elif i == 18:
    #     return 0

        # 定义每种任务的时间分布
    if i == 0:
            return 0
    elif i == 1:
            return random.uniform(2, 6)
    elif i == 2:
            return random.uniform(6, 10)
    elif i == 3:
            return random.uniform(3, 6)
    elif i == 4:
            return random.uniform(4, 9)
    elif i == 5:
            return random.uniform(6, 10)
    elif i == 6:
            return random.uniform(4, 8)
    elif i == 7:
            return random.uniform(5, 12)
    elif i == 8:
            return random.uniform(6, 10)
    elif i == 9:
            return random.uniform(4, 8)
    elif i == 10:
            return random.uniform(2, 6)
    elif i == 11:
            return random.uniform(2, 6)
    elif i == 12:
            return random.uniform(10, 16)
    elif i == 13:
            return random.uniform(6, 10)
    elif i == 14:
            return random.uniform(3, 5)

    elif i == 15:
            return random.uniform(5, 9)
    elif i == 16:
            return random.uniform(3, 5)
    elif i == 17:
            return random.uniform(4, 8)
    elif i == 18:
            return 0