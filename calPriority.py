import copy

from JZJenv.FixedMess import FixedMes
from utils import *


def calLFTandMTS(SucOrder):
    dfsLFT(SucOrder, 0)
    dfsMTS(SucOrder, 0)

def dfsLFT(SucOrder, i):
    if len(SucOrder[i].successor) == 0:
        SucOrder[i].lf = FixedMes.lowTime
        return FixedMes.lowTime

    time = 999
    for Orderid in SucOrder[i].successor:
        time = min(time,dfsLFT(SucOrder,Orderid)-SucOrder[Orderid].duration)
    SucOrder[i].lf = time
    return time

def dfsMTS(SucOrder,i):
    if len(SucOrder[i].successor)==0:
        return [SucOrder[i].id]

    record = copy.deepcopy(SucOrder[i].successor)
    for Orderid in SucOrder[i].successor:
        record = list(set(record + dfsMTS(SucOrder,Orderid)))

    SucOrder[i].mts = len(record)
    return record

#
# self.grpw = 0
# self.GRD = 0
#
# self.ACS = 0
# self.WCS = 0
def calculate_grpw(SucOrder):
    """Calculates Greatest Rank Position Wight(GRPW) for each job"""
    for i in range(FixedMes.Activity_num):
        jzjId = SucOrder[i].belong_plane_id
        taskId = SucOrder[i].taskid
        SucOrder[i].grpw = SucOrder[i].dur
        for j in SucOrder[i].successor:
            SucOrder[i].grpw += SucOrder[j].dur


def calculate_grd(SucOrder):
    for i in range(FixedMes.Activity_num):
        jzjId = SucOrder[i].belong_plane_id
        taskId = SucOrder[i].taskid
        SucOrder[i].grpw = SucOrder[i].dur
        for j in range(SucOrder[i].resourceRequestH):
            SucOrder[i].GRD += SucOrder[i].dur * SucOrder[i].resourceRequestH[j]

def calculate_dynamic_priority_rules(alltasks,eligible, current_time, current_consumption, active_list, finish_times):
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
                        earliest_start(alltasks,j, i, current_time, current_consumption, active_list, finish_times) -
                        alltasks[i].ls, irsm_val)
                    curr_e_val =earliest_start(alltasks,i, j, current_time, current_consumption, active_list, finish_times)
                    max_e_val = max(curr_e_val, max_e_val)
                    sum_e_vals += curr_e_val
            alltasks[j].irsm = irsm_val
            alltasks[j].wcs = alltasks[j].ls - max_e_val
            alltasks[j].acs = alltasks[j].ls - (1 / (len(eligible) - 1)) * sum_e_vals

def earliest_start(alltasks, i, j, current_time, current_consumption, active_list, finish_times):
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
        starts = [current_time + alltasks[i].dur]
        if isGFP(alltasks,i, j):
            pass
        elif isCSP(alltasks,i, j, current_consumption):
            starts.append(current_time)
        else:
            new_consumption = [elem for elem in current_consumption]

            new_time = current_time
            finished = [0] * (len(active_list))
            while (not isCSP(alltasks,i, j, new_consumption)):
                for act in active_list:
                    jzj = act // FixedMes.planeNum
                    task = act % FixedMes.planeOrderNum
                    if finish_times[jzj][task] == new_time and finished[active_list.index(act)] == 0:
                        finished[active_list.index(act)] = 1
                        new_consumption = sub_lists(new_consumption, alltasks[act].resourceRequestH)
                new_time += 1
            starts.append(new_time)
        return min(starts)

def isGFP(alltasks, i, j):
        """Checks if (i,j) is a Generally forbidden pair"""
        return not less_than(add_lists(alltasks[i].resourceRequestH, alltasks[j].resourceRequestH), FixedMes.total_Huamn_resource)

def isCSP(alltasks, i, j, current_consumption):
        "Checks if (i,j) is a currently schedulable pair(simultaneously)"
        new_consumption = add_lists(alltasks[i].resourceRequestH, alltasks[j].resourceRequestH)
        new_consumption = add_lists(new_consumption, current_consumption)
        return less_than(new_consumption, FixedMes.total_Huamn_resource)