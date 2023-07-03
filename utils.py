import scipy.stats as stats
from JZJenv.FixedMess import FixedMes


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
def getTime(j, i):
    # 定义每种任务的时间分布
    sigma = 0.3
    if i == 0:
        return 0
    elif i == 1:
        return 0
    elif i == 2:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[2], scale=sigma).rvs()
    elif i == 3:
        return stats.truncnorm((-0.8) / sigma, 0.8 / sigma, loc=FixedMes.OrderTime[3], scale=sigma).rvs()
    elif i == 4:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[4], scale=sigma).rvs()
    elif i == 5:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[5], scale=sigma).rvs()
    elif i == 5:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[5], scale=sigma).rvs()
    elif i == 6:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[6], scale=sigma).rvs()
    elif i == 7:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[7], scale=sigma).rvs()

    elif i == 8:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[8], scale=sigma).rvs()
    elif i == 9:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[9], scale=sigma).rvs()

    elif i == 10:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[10], scale=sigma).rvs()
    elif i == 11:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[11], scale=sigma).rvs()

    elif i == 12:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[12], scale=sigma).rvs()
    elif i == 13:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[13], scale=sigma).rvs()
    elif i == 14:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[14], scale=sigma).rvs()

    elif i == 15:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[15], scale=sigma).rvs()
    elif i == 16:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[16], scale=sigma).rvs()
    elif i == 17:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[17], scale=sigma).rvs()
    elif i == 18:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[18], scale=sigma).rvs()
    elif i == 19:
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