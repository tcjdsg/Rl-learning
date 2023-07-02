from torch.distributions.categorical import Categorical
import copy

def override(fn):
    """
    override decorator
    """
    return fn
def select_action(p, cadidate, memory):
    pri = ['LFT','LST','EST','EFT','FIFO','RAND','SPT','MTS','GRPW','GRD','IRSM','WCS','ACS']
    dist = Categorical(p.squeeze())
    s = dist.sample()
    priority_rule = pri[s]

    if (priority_rule == 'LFT'):
        return eligible[find_index(eligible, self.latest_finish_times, 'min')]
    elif (priority_rule == 'LST'):
        return eligible[find_index(eligible, self.latest_start_times, 'min')]
    elif (priority_rule == 'EST'):
        return eligible[find_index(eligible, self.earliest_start_times, 'min')]
    elif (priority_rule == 'EFT'):
        return eligible[find_index(eligible, self.earliest_finish_times, 'min')]
    elif (priority_rule == 'FIFO'):
        return sorted(eligible)[0]
    elif (priority_rule == 'RAND'):
        return random.choice(eligible)
    elif (priority_rule == 'SPT'):
        return eligible[find_index(eligible, self.durations, 'min')]
    elif (priority_rule == 'MTS'):
        return eligible[find_index(eligible, self.mts, 'max')]
    elif (priority_rule == 'GRPW'):
        return eligible[find_index(eligible, self.grpw, 'max')]
    elif (priority_rule == 'GRD'):
        return eligible[find_index(eligible, self.grd, 'max')]
    elif (priority_rule == 'IRSM'):
        return eligible[find_index(eligible, self.irsm, 'min')]
    elif (priority_rule == 'WCS'):
        return eligible[find_index(eligible, self.wcs, 'min')]
    elif (priority_rule == 'ACS'):
        return eligible[find_index(eligible, self.acs, 'min')]
    else:
        print("Invalid priority rule")

    if memory is not None: memory.logprobs.append(dist.log_prob(s))
    return cadidate[s], s

def parelle(allTasks, finished,running,finishedID,total_resource,operaNumber,AON):
    t= 0
    useNowResource = [0 for i in range(len(total_resource))]

    while True:
        #更新任务集
        D = conditionCheck(allTasks,AON,operaNumber,finishedID)
        #找到不冲突任务集W
        while True:
            W = findW(D,total_resource,running)
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

def findW(D,total_resource,P):

    W =[]
    useNowResource = [0 for i in range(len(total_resource))]

    for k in range(len(total_resource)):
        for p in P:
            useNowResource[k] = useNowResource[k] + p.resources[k]

    tempuseNowResource = copy.deepcopy(useNowResource)

    for task in D:
        for k in range(len(useNowResource)):
            if tempuseNowResource[k] > total_resource[k]:
                break
        flag = True
        for k in range(len(useNowResource)):
            if tempuseNowResource[k] + task.resources[k] > total_resource[k]:
                flag = False
                break
        if flag == True:
            W.append(task)

            for k in range(len(useNowResource)):
                tempuseNowResource[k] += task.resources[k]
    W.sort(key=lambda x: x.priority)
    return  W

def conditionCheck(allltasks,AON,code,s):
    D =[]
    for i in range(code) :
        if i in s:
            continue
        flag =True
        prenumber = AON[i] #前序
        for ordernumber in prenumber:
            if ordernumber not in s:
                flag = False
                break
        if flag == True:
            D.append(allltasks[i-1])
    return D

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
