import copy
import sys

import torch

from JZJenv.Human import Human
from JZJenv.Station import Station
from JZJenv.judge import judgeStation, allocationStation, allocationHuman
from calPriority import *
from draw.drawPeople import Draw_gantt
from utils import *
import gym
import random
import numpy as np
from Params import configs
from agent_utils import  conditionUpdateAndCheck
from  read.preprocess import InitM
from agent_utils import override
from JZJenv.FixedMess import FixedMes
np.seterr(divide='ignore',invalid='ignore')
filenameDis = "dis.csv"

device = torch.device(configs.device)
class JZJ(gym.Env):
    def __init__(self,
                 planeNum,
                 planeOrderNum):

        self.pri = ['LFT', 'LST', 'FIFO', 'RAND', 'SPT', 'MTS', 'GRPW', 'GRD', 'IRSM', 'WCS', 'ACS']

        configs.action_dim = len(self.pri)

        self.step_count = 0
        self.number_of_JZJ = planeNum
        self.number_of_opera = planeOrderNum
        self.number_of_tasks = self.number_of_JZJ * self.number_of_opera
        self.Init = InitM(filenameDis)
        FixedMes.distance = self.Init.readDis()
        self.static_activities, self.suc ,self.pre= self.Init.readData()
        calLFTandMTS(self.static_activities)

        self.finished_time =torch.tensor( [[0.0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)])
        self.scheduled_mark = torch.tensor( [[0.0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)])
        self.can_be_scheduled_mark = torch.tensor( [[0.0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)])
        self.running_tasks = []
        self.waiting_tasks = []
        self.scheduled_tasksID = []

    def done(self):
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            print(self.Cmax)
            return True
        return False

    @override
    def step(self, action):

        priority_rule = self.pri[action]
        current_time = self.t
        assert len(self.eligible) > 0
        if len(self.eligible) > 1:
            #alltasks,eligible, current_time, current_consumption, active_list, finish_times
            calculate_dynamic_priority_rules(self.activities,
                                             self.eligible,
                                             current_time,
                                             self.current_consumption,
                                             self.running_tasks)
        #基于规则选择工件
        activityIndex = self.returnActivity(priority_rule)
        print("---scheduled: {}---chooseIndex:{}---choose priority rule: {}".format(len(self.partial_sol_sequeence),activityIndex, priority_rule))

        self.can_be_scheduled_mark = torch.tensor([[0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)])

        current_time = self.t
        maxend_Time = self.Cmax

        assert activityIndex is not None and activityIndex not in self.partial_sol_sequeence

        if activityIndex is not None and activityIndex not in self.partial_sol_sequeence:

            [flag,eligibleStation] = judgeStation(self.activities, activityIndex, self.recordStation)
            assert flag==True
            # 更新信息

            self.eligible.remove(activityIndex)
            row = activityIndex // self.number_of_opera
            col = activityIndex % self.number_of_opera
            self.step_count += 1
            self.partial_sol_sequeence.append(activityIndex)

            # 更新状态，状态包括：
            self.scheduled_mark[row][col] = 1
            self.activities[activityIndex].scheduled = True
            self.activities[activityIndex].working = True
            dur_a = getTime(row, col)

            self.activities[activityIndex].es = current_time
            self.activities[activityIndex].ef = current_time + dur_a

            #分配设备和人员
            StationInfo  = allocationStation(self.activities[activityIndex], self.Stations, eligibleStation)
            allocationHuman(self.activities[activityIndex], self.Humans)

            # 记录设备工作状态
            if len(StationInfo) > 0:

                self.recordStation[StationInfo[0]][StationInfo[1]] = 1

            # 记录人员消耗
            self.current_consumption = add_lists(self.current_consumption,
                                                 self.activities[activityIndex].resourceRequestH)
            self.running_tasks.append(activityIndex)
            self.end_Time.append(current_time + dur_a)
            maxend_Time = sorted(self.end_Time, key=lambda x:x)[-1]
            eli = []
            self.t = current_time
            for i in self.eligible:
                if (less_than(self.activities[i].resourceRequestH,
                              sub_lists(FixedMes.total_Human_resource, self.current_consumption)))\
                    and judgeStation(self.activities, i, self.recordStation)[0]==True:
                    eli.append(i)

            self.eligible = eli
            if len(self.eligible) > 0:
                #如果还有可用的，就还是这个集合
                pass
            else:
                #找到新的最早结束时间
                while len(self.eligible) == 0:
                    self.t = sorted([self.activities[i].ef for i in self.running_tasks])[0]
                    current_time = self.t
                    removals =[]
                    for i in self.running_tasks:
                        if (self.activities[i].ef <= current_time):
                            self.finished.append(i)
                            self.finished_mark[i // self.number_of_opera][i % self.number_of_opera] = 1
                            self.finished_time[i // self.number_of_opera][i % self.number_of_opera] = self.activities[i].ef
                            self.activities[i].working = False
                            self.activities[i].complete = True
                            self.current_consumption = sub_lists(self.current_consumption,
                                                             self.activities[i].resourceRequestH)



                            if self.activities[i].RequestStationType >= 0:
                                typeS = self.activities[i].SheiBei[0][0]
                                index = self.activities[i].SheiBei[0][1]
                                self.recordStation[typeS][index] = 0
                                self.Stations[typeS][index].working = False


                            for infoHuman in self.activities[i].HumanNums:
                                type = infoHuman[0]
                                index = infoHuman[1]
                                self.Humans[type][index].working = False

                            removals.append(i)


                    for i in removals:
                        self.running_tasks.remove(i)

                    if (len(self.finished) == FixedMes.Activity_num):
                            break

                    self.eligible = conditionUpdateAndCheck(self.activities,
                                                    self.current_consumption,
                                                    self.finished,
                                                    self.partial_sol_sequeence,
                                                    self.recordStation)

            for num in self.eligible:
                    row = num // self.number_of_opera
                    col = num % self.number_of_opera
                    self.can_be_scheduled_mark[row][col] = 1

        if torch.max(self.finished_time) == 0:
            fea = torch.stack((self.finished_time , self.scheduled_mark, self.can_be_scheduled_mark),dim=0)
        else:
            fea = torch.stack((self.finished_time/torch.max(self.finished_time), self.scheduled_mark, self.can_be_scheduled_mark),dim=0)
        reward = - (maxend_Time - self.Cmax)
        if reward == 0:
            reward = configs.rewardscale
            self.posRewards += reward
        self.Cmax = maxend_Time
        return fea.to(device), reward, self.done(), self.Cmax

    def returnActivity(self, priority_rule):
        value_lists = []

        if (priority_rule == 'LFT'):
            for i in self.eligible:
                value_lists.append(self.activities[i].lf)
            return self.eligible[find_index(self.eligible, value_lists, 'min')]
        elif (priority_rule == 'LST'):
            for i in self.eligible:
                value_lists.append(self.activities[i].ls)
            return self.eligible[find_index(self.eligible, value_lists, 'min')]

        elif (priority_rule == 'FIFO'):
            return sorted(self.eligible)[0]
        elif (priority_rule == 'RAND'):
            return random.choice(self.eligible)
        elif (priority_rule == 'SPT'):
            for i in self.eligible:
                value_lists.append(self.activities[i].duration)
            return self.eligible[find_index(self.eligible, value_lists, 'min')]
        elif (priority_rule == 'MTS'):
            for i in self.eligible:
                value_lists.append(self.activities[i].mts)
            return self.eligible[find_index(self.eligible, value_lists, 'max')]
        elif (priority_rule == 'GRPW'):
            for i in self.eligible:
                value_lists.append(self.activities[i].grpw)
            return self.eligible[find_index(self.eligible, value_lists, 'max')]
        elif (priority_rule == 'GRD'):
            for i in self.eligible:
                value_lists.append(self.activities[i].grd)
            return self.eligible[find_index(self.eligible, value_lists, 'max')]
        elif (priority_rule == 'IRSM'):
            for i in self.eligible:
                value_lists.append(self.activities[i].irsm)
            return self.eligible[find_index(self.eligible, value_lists, 'min')]
        elif (priority_rule == 'WCS'):
            for i in self.eligible:
                value_lists.append(self.activities[i].wcs)
            return self.eligible[find_index(self.eligible, value_lists, 'min')]
        elif (priority_rule == 'ACS'):
            for i in self.eligible:
                value_lists.append(self.activities[i].acs)
            return self.eligible[find_index(self.eligible, value_lists, 'min')]
        else:
            print("Invalid priority rule")

    def init(self):
        self.Humans = []
        self.Stations = []
        number = 0
        for i in range(FixedMes.Human_resource_type):
                self.Humans.append([])
                for j in range(FixedMes.total_Human_resource[i]):
                    # ij都是从0开头 ,number也是
                    self.Humans[i].append(Human([i, j, number]))
                    number += 1


        number = 0
        for i in range(FixedMes.station_resource_type):
                self.Stations.append([])
                for j in range(FixedMes.total_station_resource[i]):
                    # ij都是从0开头 ,number也是
                    self.Stations[i].append(Station([i, j, number]))
                    number += 1
    @override
    def reset(self):

        #人员设备初始化
        self.init()

        self.t = 0
        for i, activity in self.static_activities.items():

            #随机生成入场时间
            if activity.taskid==0:
                activity.ST = 0
            activity.es = 0
            activity.ef = 0
            activity.acs = 0
            activity.wcs = 0
            activity.irsm = 0

            activity.scheduled = False
            activity.working = False
            activity.complete = False

            activity.HumanNums = []  # 执行任务的人员编号
            activity.SheiBei = []

        self.activities = self.static_activities
        self.finished_mark = [[0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)]

        self.finished_time =torch.tensor([[0.0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)])
        self.scheduled_mark = torch.tensor([[0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)])
        self.can_be_scheduled_mark = torch.tensor([[0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)])

        self.running_tasks = []
        self.end_Time = []
        self.finished = []
        self.step_count = 0
        # record action history
        self.partial_sol_sequeence = []
        self.recordStation = [[0 for _ in range(FixedMes.total_station_resource[i])]
                              for i in range(len(FixedMes.total_station_resource))]


        self.posRewards = 0
        self.initQuality=0
        self.Cmax = 0
        self.current_consumption = [0 for _ in range(len(FixedMes.total_Human_resource))]

        fea = torch.stack((self.finished_time , self.scheduled_mark, self.can_be_scheduled_mark),dim=0)

        #allltasks,current_consumption,running,allNums,finished
        self.eligible = conditionUpdateAndCheck(self.static_activities,
                                                self.current_consumption,
                                                self.finished,
                                                self.partial_sol_sequeence,
                                                self.recordStation)

        return fea.to(device)

if __name__ == '__main__':
    env = JZJ(configs.n_j, configs.n_m)
    env.reset()
    for i in range(configs.n_j* configs.n_m):

        env.step(3)
    Draw_gantt(env.Humans)

