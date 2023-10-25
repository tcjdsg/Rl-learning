from torch.distributions.categorical import Categorical
import copy
from Params import configs
from JZJenv.FixedMess import FixedMes
from JZJenv.judge import judgeStation
from calPriority import calculate_dynamic_priority_rules
from utils import *

def override(fn):
    """
    override decorator
    """
    return fn
def select_action(p, eligible, memory):
    pri = ['LFT','LST','EST','EFT','FIFO','RAND','SPT','MTS','GRPW','GRD','IRSM','WCS','ACS']
    dist = Categorical(p.squeeze())
    s = dist.sample()

    if memory is not None:
        memory.logprobs.append(dist.log_prob(s))
    return s

def conditionUpdateAndCheck(n_orders, pre, exist, current_consumption, finished_mark, partitial, Stations):
    precedence_eligible = []# 满足紧前工序已完成的工序
    eligible = []# 满足紧前工序已结束的工序中满足资源约束的工序
    for i in range(len(finished_mark)):
        if i in partitial or finished_mark[i]==1 or exist[i]==0:
            continue
        flag = True
        prenumber = pre[i]# 前序
        for ordernumber in prenumber:
            if finished_mark[ordernumber] == 0:
                flag = False
                break
        if flag == True:
            precedence_eligible.append(i)
    for i in precedence_eligible:
        row = i // n_orders
        col = i % n_orders
        consump = [FixedMes.OrderInputMes[col][0][1] if i == FixedMes.OrderInputMes[col][0][0] else 0 for i in
                    range(configs.Human_resource_type)]

        if (less_than(consump, sub_lists(configs.total_Human_resource, current_consumption))) and judgeStation(row, col,Stations)[0]:
            eligible.append(i)
    return eligible

# evaluate the actions
def eval_actions(p, actions):
    softmax_dist = Categorical(p)
    ret = softmax_dist.log_prob(actions).reshape(-1)
    entropy = softmax_dist.entropy().mean()
    return ret, entropy


# select action method for test
def greedy_select_action(p, candidate):
    _, index = p.squeeze().max(0)
    action = candidate[index]
    return action


# select action method for test
def sample_select_action(p, candidate):
    dist = Categorical(p.squeeze())
    s = dist.sample()
    return candidate[s]
