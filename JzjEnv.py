import pandas as pd
import torch

from JZJenv.Human import Human
from JZJenv.Station import Station
from JZJenv.judge import judgeStation, allocationStation, allocationHuman
from calPriority import *
from draw.drawPeople import Draw_gantt
from genData import getData
from utils import *
import gym
import random
import numpy as np
from Params import configs
from agent_utils import  conditionUpdateAndCheck
from agent_utils import override
from JZJenv.FixedMess import FixedMes
np.seterr(divide='ignore',invalid='ignore')
filenameDis = "dis.csv"

device = torch.device(configs.device)
class JZJ(gym.Env):
    def __init__(self, adj, sucActDict, preActDict, time, exist):
        #self.recordWorkStation = [[0, 0] for _ range()]
        self.adj = adj
        self.step_count = 0
        self.number_of_JZJ = configs.n_j
        self.number_of_opera = configs.n_m
        self.number_of_tasks = self.number_of_JZJ * self.number_of_opera
        self.recordWorkStation = [[] for _ in range(self.number_of_tasks)]
        self.recordWorkHuman = [[] for _ in range(self.number_of_tasks)]
        self.suc = sucActDict
        self.pre = preActDict
        self.time = time
        self.exist = exist
        self.distance = FixedMes.distance
        self.n_humans = configs.total_Huamn_resource
        self.number_of_humans = sum(self.n_humans)
        self.allLei = len(self.n_humans)
        self.human_actions = HumanActions(self.number_of_humans)
        self.h_index = [0]
        for i in range(self.allLei-1):
            self.h_index.append(self.h_index[-1] + self.n_humans[i])

        calLFTandMTS(self.activities)
    def done(self):
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            print(self.Cmax)
            return True
        return False

    @override
    def step(self, action, human_a):#是否再来一次
        current_time = self.t
        reward=0

        assert len(self.eligible) > 0
        # if len(self.eligible) > 1:
        #     #alltasks,eligible, current_time, current_consumption, active_list, finish_times
        #     calculate_dynamic_priority_rules(self.activities,
        #                                      self.eligible,
        #                                      current_time,
        #                                      self.current_consumption,
        #                                      self.running_tasks)
        #基于规则选择工件
        #activityIndex = self.returnActivity(priority_rule)

        activityIndex = action

        row = activityIndex // self.number_of_opera
        col = activityIndex % self.number_of_opera

        type = FixedMes.OrderInputMes[col][0][0]

        human1 = self.human_actions[human_a][0]
        human_index1 = human1 - self.h_index[type]
        human2 = self.human_actions[human_a][1]
        human_index2 = human2 - self.h_index[type]

        assert activityIndex is not None and activityIndex not in self.partial_sol_sequeence
        if activityIndex is not None and activityIndex not in self.partial_sol_sequeence:

            [flag, eligibleStation] = judgeStation(row, col, self.Stations)
            assert flag == True
            # 更新信息

            self.step_count += 1
            dur_a = self.time[activityIndex]

            human_pos1 = self.Humans[type][human_index1].NowJZJ
            human_pos2 = self.Humans[type][human_index2].NowJZJ

            self.recordWorkHuman[activityIndex].append([type,human_index1])
            if human_index1!=human_index2:
                self.recordWorkHuman[activityIndex].append([type, human_index2])

            to_pos = row
            moveTime1 = self.distance[human_pos1][to_pos] / FixedMes.human_walk_speed
            moveTime2 = self.distance[human_pos2][to_pos] / FixedMes.human_walk_speed
            moveTime = max(moveTime1, moveTime2)
            self.can_workTime[human1] = current_time + moveTime + dur_a
            self.can_workTime[human2] = current_time + moveTime + dur_a
            self.h_workTime[human1] += dur_a
            self.h_workTime[human2] += dur_a
            self.h_working[human1] = 1
            self.h_working[human2] = 1

            # print("---scheduled: {}---chooseIndex:{}".format(len(self.partial_sol_sequeence),activityIndex//self.number_of_JZJ))

            self.can_be_scheduled_mark = torch.tensor(
                [[0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)])
            current_time = self.t

            self.eligible.remove(activityIndex)
            self.partial_sol_sequeence.append(activityIndex)
                # 更新状态，状态包括：
            self.scheduled_mark[activityIndex] = 1
                # 按规则分配设备

            StationInfo = allocationStation(self.number_of_opera, activityIndex, self.Stations, eligibleStation)
                # 记录设备工作状态
            if len(StationInfo) > 0:
                    self.recordWorkStation[activityIndex].append(StationInfo)
                    self.Stations[StationInfo[0]][StationInfo[1]].working = True
            self.running_tasks.append(activityIndex)

            self.LB[activityIndex].es = current_time + moveTime
            self.LB[activityIndex].ef = current_time + moveTime + dur_a

            for i in range(self.number_of_humans):
                if self.can_workTime[i] < current_time:
                    self.can_workTime[i] = current_time

            # 记录人员消耗
            nowConsump = [FixedMes.OrderInputMes[col][0][1] if i == FixedMes.OrderInputMes[col][0][0] else 0 for i in range(configs.Human_resource_type)]

            self.current_consumption = add_lists(self.current_consumption,nowConsump)
            eli = []
            self.t = current_time
            for i in self.eligible:
                if (less_than(nowConsump, sub_lists(FixedMes.total_Human_resource, self.current_consumption)))\
                    and judgeStation(i//self.number_of_opera, i%self.number_of_opera, self.Stations)[0] == True:
                    eli.append(i)

            self.eligible = eli
            if len(self.eligible) > 0:
                #如果还有可用的，就还是这个集合
                pass
            else:
                #找到新的最早结束时间
                while len(self.eligible) == 0:
                    self.t = sorted([self.LB[i] for i in self.running_tasks])[0]
                    current_time = self.t
                    for i in range(self.number_of_humans):
                        # 如果当前时间大于人员最早可用时间，则修改人员可用时间为当前时间，人员状态为空闲
                        if self.can_workTime[i] < current_time:
                            self.h_working[i] = 0
                            self.can_workTime[i] = current_time

                    removals =[]
                    for i in self.running_tasks:
                        if (self.LB[i] <= current_time and self.scheduled_mark[i] == 1):
                            self.finished.append(i)
                            self.finished_mark[i] = 1
                            #self.LB[i] = self.activities[i].ef # 修改工序完成下限为实际完成时间
                            humanType = FixedMes.OrderInputMes[i%self.number_of_opera][0][0]
                            humanNeed = FixedMes.OrderInputMes[i%self.number_of_opera][0][1]
                            self.current_consumption = sub_lists(self.current_consumption,
                                                                 [humanNeed if i == humanType else 0 for i in range(configs.Human_resource_type)])

                            if FixedMes.OrderInputMes[i % self.number_of_opera][1][0] >= 0:
                                typeS = self.recordWorkStation[i][0][0]
                                index = self.recordWorkStation[i][0][1]
                                self.Stations[typeS][index].working = False

                            removals.append(i)
                    for i in removals:
                        self.running_tasks.remove(i)

                    if (len(self.finished) == self.number_of_tasks):
                            break

                    self.eligible = conditionUpdateAndCheck(self.number_of_opera,
                                                            self.pre,
                                                            self.exist,
                                                            self.current_consumption,
                                                            self.finished,
                                                            self.partial_sol_sequeence,
                                                            self.Stations)

            for num in self.eligible:

                    self.can_be_scheduled_mark[num] = 1

            dfsEF(self.pre, self.scheduled_mark, self.number_of_tasks, self.LB)  # 计算每个工序的最早完成时间
            self.end_Time.append(current_time + moveTime + dur_a)
            maxend_Time = sorted(self.end_Time, key=lambda x: x)[-1]

            reward = - (maxend_Time - self.Cmax)
            if reward == 0:
                reward = configs.rewardscale
                self.posRewards += reward
            self.Cmax = maxend_Time

        ''' 工序特征 '''
        fea = torch.stack((self.LB/configs.et_normalize_coef, self.scheduled_mark,self.can_be_scheduled_mark), dim=0)
        ''' 人员特征 '''
        fea_h = torch.stack((self.h_workTime/configs.et_normalize_coef, self.h_working, self.can_workTime/configs.et_normalize_coef),dim=0)

        self.task_mask = [1 if i not in self.eligible else 0 for i in range(self.number_of_tasks)]
        self.human_mask = [1 if time > current_time else 0 for time in self.can_workTime]

        return fea.to(device), fea_h.to(device), reward, self.done(), self.Cmax, self.task_mask, self.human_mask
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
        for i in range(configs.Human_resource_type):
                self.Humans.append([])
                for j in range(self.n_humans[i]):
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


        self.finished_mark = [0 for _ in range(self.number_of_tasks)]
        self.LB = torch.tensor([0.0 for _ in range(self.number_of_tasks)])
        self.scheduled_mark = torch.tensor([0.0 for _ in range(self.number_of_opera * self.number_of_JZJ)])
        self.can_be_scheduled_mark = torch.tensor([0.0 for _ in range(self.number_of_opera * self.number_of_JZJ)])
        dfsEF(self.pre, self.scheduled_mark, self.number_of_tasks, self.LB)  # 计算 self.LB

        self.running_tasks = []
        self.end_Time = []
        self.finished = []
        self.step_count = 0
        # record action history
        self.partial_sol_sequeence = []

        #表示工作状态
        self.stateStation = [[0 for _ in range(FixedMes.total_station_resource[i])]
                              for i in range(len(FixedMes.total_station_resource))]
        self.stateHuman = [[0 for _ in range(configs.total_Huamn_resource[i])]
                              for i in range(len(configs.Human_resource_type))]

        self.posRewards = 0
        self.initQuality=0
        self.Cmax = 0
        self.current_consumption = [0 for _ in range(len(FixedMes.total_Human_resource))]

        fea = torch.stack((self.LB, self.scheduled_mark, self.can_be_scheduled_mark), dim=0)

        self.h_workTime = torch.tensor([0.0 for _ in self.number_of_humans])
        self.h_working = torch.tensor([0 for _ in self.number_of_humans])
        self.can_workTime = torch.tensor([0.0 for _ in self.number_of_humans])

        fea_h = torch.stack((self.h_workTime, self.h_working, self.can_workTime), dim=0)

        #allltasks,current_consumption,running,allNums,finished
        self.eligible = conditionUpdateAndCheck(self.number_of_opera,
                                                self.pre,
                                                self.exist,
                                                self.current_consumption,
                                                self.finished,
                                                self.partial_sol_sequeence,
                                                self.Stations)

        self.mask_order = np.full(shape=(self.number_of_JZJ * self.number_of_opera), fill_value=0, dtype=bool)
        self.mask_human = np.full(shape=(self.number_of_JZJ * self.number_of_opera), fill_value=0, dtype=bool)

        return fea.to(device), self.adj.to(device), fea_h.to(device),

if __name__ == '__main__':
    adj, sucActDict, preActDict, time, exist = getData(configs.n_jzj, configs.n_order, configs.n_human)
    env = JZJ(adj, sucActDict, preActDict, time, exist)
    a,b,c = env.reset()
    for i in range(configs.n_j* configs.n_m):

        env.step(3)
    Draw_gantt(env.Humans)

