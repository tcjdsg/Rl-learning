import scipy.stats as stats

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

def getTime(j, i):

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


import numpy as np


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

