from torch.distributions.categorical import Categorical
import copy

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

def conditionUpdateAndCheck(allltasks,current_consumption,finished,partitial,recordStation):

    # 满足紧前工序已完成的工序
    precedence_eligible =[]
    # 满足紧前工序已结束的工序中满足资源约束的工序
    eligible = []

    for i in range(FixedMes.Activity_num):
        if i in partitial or i in finished:
            continue

        flag = True
        prenumber = allltasks[i].predecessor #前序
        for ordernumber in prenumber:
            if ordernumber not in finished:
                flag = False
                break
        if flag == True:
            precedence_eligible.append(allltasks[i].id)

    for i in precedence_eligible:
        if (less_than(allltasks[i].resourceRequestH, sub_lists(FixedMes.total_Human_resource, current_consumption)))\
                and judgeStation(allltasks,i,recordStation)[0]==True:
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
