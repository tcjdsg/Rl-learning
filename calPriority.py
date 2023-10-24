import copy

from JZJ_PPO.JZJenv.FixedMess import FixedMes
from utils import *


def calLFTandMTS(SucOrder):
    for i in range(FixedMes.planeNum):
        dfsLFT(SucOrder, i*FixedMes.planeOrderNum)
        dfsMTS(SucOrder, i*FixedMes.planeOrderNum)
    calculate_grpw(SucOrder)
    calculate_grd(SucOrder)

def dfsLFT(SucOrder, i):
    if len(SucOrder[i].successor) == 0:
        SucOrder[i].lf = FixedMes.lowTime
        return FixedMes.lowTime
    time = 999
    for Orderid in SucOrder[i].successor:
        time = min(time, dfsLFT(SucOrder, Orderid) - SucOrder[Orderid].duration)
    SucOrder[i].lf = time
    SucOrder[i].ls = time - SucOrder[i].duration
    return time

def dfsEF(activities, i, LB):
    if len(activities[i].precessor) == 0:
        activities[i].es = 0
        activities[i].ef = 0
        LB.append(0)
        return 0
    time = 0
    for Orderid in activities[i].pre:
        time = max(time, dfsEF(activities, Orderid, LB))
    if ~activities[i].scheduled:
        activities[i].es = time
        activities[i].ef = time + activities[i].duration
    LB.append(activities[i].ef)
    return activities[i].ef

def dfsMTS(SucOrder,i):
    if len(SucOrder[i].successor) == 0:
        return [SucOrder[i].id]
    record = copy.deepcopy(SucOrder[i].successor)
    for Orderid in SucOrder[i].successor:
        record = list(set(record + dfsMTS(SucOrder, Orderid)))

    SucOrder[i].mts = len(record)
    return record


def calculate_grpw(SucOrder):
    """Calculates Greatest Rank Position Wight(GRPW) for each job"""
    for i in range(FixedMes.Activity_num):
        jzjId = SucOrder[i].belong_plane_id
        taskId = SucOrder[i].taskid
        SucOrder[i].grpw = SucOrder[i].duration
        for j in SucOrder[i].successor:
            SucOrder[i].grpw += SucOrder[j].duration

def calculate_grd(SucOrder):
    for i in range(FixedMes.Activity_num):
        jzjId = SucOrder[i].belong_plane_id
        taskId = SucOrder[i].taskid
        SucOrder[i].grd = SucOrder[i].duration
        for j in range(len(SucOrder[i].resourceRequestH)):
            SucOrder[i].grd += SucOrder[i].duration * SucOrder[i].resourceRequestH[j]

def calculate_dynamic_priority_rules(alltasks,eligible, current_time, current_consumption, active_list):
        """
            Calculates IRSM, WCS, ACS priority values


            Parameters:
                eligible: eligible set of jobs based on both precedence and resource constraints
                current_time: Current time when priorities are being calculated
                current_consumption: Amount of resources being consumed currently
                active_list: List of jobs which are scheduled and currently active
                finish_times: Finish times of each job
        """
        for j in eligible:
            sum_e_vals = 0  # Sum of E(i,j) over all i
            max_e_val = 0  # Max of E(i,j) over all i
            irsm_val = 0  # Max of max(0,E(j,i) -LS_i) over all i
            for i in eligible:
                if (i != j):
                    irsm_val = max(
                        earliest_start(alltasks,j, i, current_time, current_consumption, active_list) -
                        alltasks[i].ls, irsm_val)
                    curr_e_val = earliest_start(alltasks,i, j, current_time, current_consumption, active_list)
                    max_e_val = max(curr_e_val, max_e_val)
                    sum_e_vals += curr_e_val
            alltasks[j].irsm = irsm_val
            alltasks[j].wcs = alltasks[j].ls - max_e_val
            alltasks[j].acs = alltasks[j].ls - (1 / (len(eligible) - 1)) * sum_e_vals

def earliest_start(alltasks, i, j, current_time, current_consumption, active_list):
        """
            Find's the earliest time j can be scheduled if i is scheduled at current_time

            Parameters:
                i,j : Jobs
                current_time: Current time when priorities are being calculated
                current_consumption: Amount of resources being consumed currently
                active_list: List of jobs which are scheduled and currently active
                finish_times: Finish times of each job
            Returns:
                E(i,j)
        """
        starts = [current_time + alltasks[i].duration]
        if isGFP(alltasks,i, j):
            pass
        elif isCSP(alltasks,i, j, current_consumption):
            starts.append(current_time)
        else:
            new_consumption = [elem for elem in current_consumption]

            new_time = round(current_time,1)
            finished = [0] * (len(active_list))
            while (not isCSP(alltasks,i, j, new_consumption)):
                for act in active_list:
                    jzj = act // FixedMes.planeOrderNum
                    task = act % FixedMes.planeOrderNum
                    if round(alltasks[act].ef, 1) == round(new_time,1) and finished[active_list.index(act)] == 0:
                        finished[active_list.index(act)] = 1
                        new_consumption = sub_lists(new_consumption, alltasks[act].resourceRequestH)
                new_time += 0.1
            starts.append(new_time)
        return min(starts)

def isGFP(alltasks, i, j):
        """Checks if (i,j) is a Generally forbidden pair"""
        return not less_than(add_lists(alltasks[i].resourceRequestH, alltasks[j].resourceRequestH), FixedMes.total_Human_resource)

def isCSP(alltasks, i, j, current_consumption):
        "Checks if (i,j) is a currently schedulable pair(simultaneously)"
        new_consumption = add_lists(alltasks[i].resourceRequestH, alltasks[j].resourceRequestH)
        new_consumption = add_lists(new_consumption, current_consumption)
        return less_than(new_consumption, FixedMes.total_Human_resource)


