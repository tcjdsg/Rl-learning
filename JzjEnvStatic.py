import pandas as pd
import torch

from JZJenv.Human import Human
from JZJenv.Station import Station
from JZJenv.Space import Space
from JZJenv.judge import *
from calPriority import *
from calculateLB import *
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
        self.number_of_JZJ = configs.n_jzjs
        self.number_of_opera = configs.n_orders
        self.number_of_tasks = self.number_of_JZJ * self.number_of_opera
        self.recordWorkStation = [[] for _ in range(self.number_of_tasks)]
        self.recordWorkHuman = [[] for _ in range(self.number_of_tasks)]
        self.suc = sucActDict
        self.pre = preActDict
        self.time = time
        self.exist = exist
        self.distance = FixedMes.distance
        self.n_humans = configs.total_Human_resource
        self.number_of_humans = sum(self.n_humans)
        self.allLei = len(self.n_humans)
        self.human_actions,self.human_a_index = HumanActions(self.n_humans)
        self.h_index = [0]
        for i in range(self.allLei-1):
            self.h_index.append(self.h_index[-1] + self.n_humans[i])

        #calLFTandMTS(self.activities)
    def done(self):
        if len(self.partial_sol_sequeence) == sum(self.exist):
            print(self.Cmax)
            return True
        return False

    @override
    def step(self, action, human_a):
        current_time = self.t
        reward=0

        assert len(self.eligible) > 0
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

            #[flag, eligibleStation] = judgeStation(row, col, self.Stations)

            #assert flag == True
            # 更新信息

            self.step_count += 1
            dur_a = self.time[activityIndex]

            human_pos1 = self.Humans[type][human_index1].NowJZJ
            human_pos2 = self.Humans[type][human_index2].NowJZJ

            self.Humans[type][human_index1].update(activityIndex,dur_a)
            self.Humans[type][human_index1].workig = True
            self.recordWorkHuman[activityIndex].append([type,human_index1])

            self.h_workTime[human1] += dur_a

            if human_index1 != human_index2:
                self.Humans[type][human_index2].update(activityIndex, dur_a)
                self.Humans[type][human_index2].workig = True
                self.recordWorkHuman[activityIndex].append([type, human_index2])
                self.h_workTime[human2] += dur_a


            # print("---scheduled: {}---chooseIndex:{}".format(len(self.partial_sol_sequeence),activityIndex//self.number_of_JZJ))

            self.can_be_scheduled_mark = torch.tensor([0 for _ in range(self.number_of_tasks)])
            current_time = self.t

            self.eligible.remove(activityIndex)
            self.partial_sol_sequeence.append(activityIndex)
                # 更新状态，状态包括：
            self.scheduled_mark[activityIndex] = 1
                # 按规则分配设备
            StationInfo = allocationStationStatic(self.number_of_opera, activityIndex, self.Stations,  dur_a)
                # 记录设备工作状态
            if len(StationInfo) > 0:
                    self.recordWorkStation[activityIndex].append(StationInfo)
                    self.Stations[StationInfo[0]][StationInfo[1]].working = True

            if FixedMes.OrderInputMes[col][2][1] == 1:
                self.Spaces[row].working = True
                self.Spaces[row].update(activityIndex)

            self.running_tasks.append(activityIndex)
            self.adj,self.LB, self.can_workTime ,self.dis_time = calculateLBs(self.Humans, self.Stations, self.Spaces, self.suc, self.time)

            self.eligible = condition_check(self.pre, self.exist, self.scheduled_mark, self.partial_sol_sequeence)

            maxend_Time = sorted(self.LB, key=lambda x: x)[-1]

            reward = - (maxend_Time - self.Cmax)
            if reward == 0:
                reward = configs.rewardscale
                self.posRewards += reward
            self.Cmax = maxend_Time

        ''' 工序特征 '''
        self.LBm = self.LB.reshape(-1, 1)
        self.scheduled_markm = self.scheduled_mark.reshape(-1, 1)
        fea = np.concatenate((self.LBm, self.scheduled_mark), axis=-1)

        ''' 人员特征 '''
        fea_h = torch.stack((self.h_workTime/configs.et_normalize_coef, self.dis_time/configs.et_normalize_coef, self.can_workTime/configs.et_normalize_coef),dim=0)



        self.task_mask = [1 if i not in self.eligible else 0 for i in range(self.number_of_tasks)]
        # self.human_mask = [1 if time > current_time else 0 for time in self.can_workTime]

        return self.adj, fea, fea_h.to(device), reward, self.done(), self.Cmax, self.task_mask, self.h_index

    def init(self):
        self.Humans = []
        self.Stations = []
        self.Spaces = []

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

        for i in range(self.number_of_JZJ):
                self.Spaces.append(Space(i))


    @override
    def reset(self):
        #人员设备初始化
        self.init()
        self.t = 0
        self.finished_mark = [0 for _ in range(self.number_of_tasks)]
        self.LB =  np.array([0.0 for _ in range(self.number_of_tasks)])
        self.scheduled_mark = np.array([0.0 for _ in range(self.number_of_tasks)])
        self.can_be_scheduled_mark =  np.array([0.0 for _ in range(self.number_of_tasks)])

        dfsEF(self.pre, self.scheduled_mark, self.number_of_tasks-1, self.LB)  # 计算 self.LB
        self.running_tasks = []
        self.end_Time = []
        self.step_count = 0
        # record action history
        self.partial_sol_sequeence = []
        self.posRewards = 0
        self.initQuality=0
        self.Cmax = 0
        #allltasks,current_consumption,running,allNums,finished
        self.LB, self.can_workTime, self.dis_time = calculateLBs(self.Humans, self.Stations, self.Spaces, self.suc,
                                                                 self.time)
        self.LBm = self.LB.reshape(-1, 1)
        self.scheduled_markm = self.scheduled_mark.reshape(-1, 1)
        fea = np.concatenate((self.LBm, self.scheduled_mark), axis=-1)

        self.h_workTime = np.concatenate([0.0 for _ in range(self.number_of_humans)])
        self.dis_time = np.concatenate(self.dis_time)
        self.can_workTime = np.concatenate(self.can_workTime)

        fea_h = np.concatenate((self.h_workTime, self.dis_time, self.can_workTime), axis=-1)

        self.eligible = condition_check(self.pre,self.exist,self.scheduled_mark,self.partial_sol_sequeence)



        self.task_mask = [1 if i not in self.eligible else 0 for i in range(self.number_of_tasks)]
        #TODO human_mask


        return fea, self.adj, fea_h ,self.h_index

if __name__ == '__main__':
    adj, sucActDict, preActDict, time, exist = getData(configs.n_jzjs, configs.n_orders, configs.total_Human_resource)
    env = JZJ(adj, sucActDict, preActDict, time, exist)
    a,b,c = env.reset()
    acs, index = HumanActions(configs.total_Human_resource)
    print(sum(exist))

    for i in range(sum(exist)):
        print(i)
        type = FixedMes.OrderInputMes[env.eligible[0]%configs.n_orders][0][0]

        env.step(env.eligible[0],index[type])

    Draw_gantt(env.Humans)

