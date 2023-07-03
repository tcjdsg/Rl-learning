from torch.distributions.categorical import Categorical
import copy

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

def parelle(allTasks, finished,running,finishedID,total_resource,operaNumber,AON):
    t= 0
    useNowResource = [0 for i in range(len(total_resource))]

    while True:
        #更新任务集
        D = conditionCheck(allTasks,AON,operaNumber,finishedID)
        #找到不冲突任务集W
        while True:
            W = findW(allTasks,D,total_resource,running)
            if len(W) > 0:
                taskj = W[0]
                taskj.es = t
                taskj.ef = t + taskj.dur
                running.append(taskj)
                D.remove(taskj)
                finishedID.append(taskj.id)

            else:
                break

        running.sort(key = lambda x:x.ef)
        taski = running[0]
        finished.append(taski)
        t = taski.f
        running.remove(taski)
        for otherp in running:
            if otherp.f == t:
                running.remove(otherp)
                finished.append(otherp)
        if len(finished)==len(allTasks):
            break

def findW(allTasks,D,total_resource,P):

    W =[]
    useNowResource = [0 for i in range(len(total_resource))]

    for k in range(len(total_resource)):
        for p in P:
            useNowResource[k] = useNowResource[k] + p.resources[k]

    tempuseNowResource = copy.deepcopy(useNowResource)

    for task in D:
        flag = True
        for k in range(len(useNowResource)):
            if tempuseNowResource[k] + allTasks[task].resources[k] > total_resource[k]:
                flag = False
                break
        if flag == True:
            W.append(task)

            for k in range(len(useNowResource)):
                tempuseNowResource[k] += allTasks[task].resources[k]
    W.sort(key=lambda x: x.priority)
    return  W

def conditionUpdateAndCheck(allltasks,current_consumption,finished):


    # 满足紧前工序已完成的工序
    precedence_eligible =[]
    # 满足紧前工序已结束的工序中满足资源约束的工序
    eligible = []

    for i in range(FixedMes.Activity_num):
        if i in finished:
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
        if (less_than(allltasks[i].resourceRequestH, sub_lists(FixedMes.total_Huamn_resource, current_consumption))):
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