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
    def __init__(self, data):

        self.step_count = 0
        self.number_of_JZJ = data['planeNum']
        self.number_of_opera = data['planeOrderNum']
        self.number_of_tasks = self.number_of_JZJ * self.number_of_opera
        self.data = data
        self.distance = data['distance']
        self.suc = data['suc']
        self.pre = data['pre']
        self.n_humans = data['human']
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
        task = self.activities[activityIndex]
        type = task.resourceRequestH

        human1 = self.human_actions[human_a][0]
        human_index1 = human1 - self.h_index[type]
        human2 = self.human_actions[human_a][1]
        human_index2 = human2 - self.h_index[type]



        assert activityIndex is not None and activityIndex not in self.partial_sol_sequeence
        if activityIndex is not None and activityIndex not in self.partial_sol_sequeence:

            [flag, eligibleStation] = judgeStation(self.activities, activityIndex, self.recordStation)
            assert flag == True
            # 更新信息
            row = activityIndex // self.number_of_opera
            col = activityIndex % self.number_of_opera
            self.step_count += 1
            dur_a = getTime(row, col)
            self.activities[activityIndex].dur = dur_a

            human_pos1 = self.Humans[type][human_index1].NowJZJ
            human_pos2 = self.Humans[type][human_index2].NowJZJ

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

            task.HumanNums.append([type, human_index1])
            if human_index1!=human_index2:
                task.HumanNums.append([type, human_index2])
            # print("---scheduled: {}---chooseIndex:{}".format(len(self.partial_sol_sequeence),activityIndex//self.number_of_JZJ))

            self.can_be_scheduled_mark = torch.tensor(
                [[0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)])
            current_time = self.t

            self.eligible.remove(activityIndex)
            self.partial_sol_sequeence.append(activityIndex)
                # 更新状态，状态包括：
            self.scheduled_mark[row][col] = 1
            task.scheduled = True
            task.working = True
                # 按规则分配设备
            StationInfo = allocationStation(self.activities[activityIndex], self.Stations, eligibleStation)
                # 记录设备工作状态
            if len(StationInfo) > 0:
                    self.recordStation[StationInfo[0]][StationInfo[1]] = 1
            self.running_tasks.append(activityIndex)


            self.activities[activityIndex].es = current_time+moveTime
            self.activities[activityIndex].ef = current_time+moveTime + dur_a



            for i in range(self.number_of_humans):
                if self.can_workTime[i] < current_time:
                    self.can_workTime[i] = current_time

            # 记录人员消耗
            nowConsump = [1 if i==self.activities[activityIndex].resourceRequestH
                                                  else 0 for i in range(FixedMes.Human_resource_type)]

            self.current_consumption = add_lists(self.current_consumption,nowConsump)
            eli = []
            self.t = current_time
            for i in self.eligible:
                if (less_than(nowConsump, sub_lists(FixedMes.total_Human_resource, self.current_consumption)))\
                    and judgeStation(self.activities, i, self.recordStation)[0] == True:
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
                    for i in range(self.number_of_humans):
                        # 如果当前时间大于人员最早可用时间，则修改人员可用时间为当前时间，人员状态为空闲
                        if self.can_workTime[i] < current_time:
                            self.h_working[i] = 0
                            self.can_workTime[i] = current_time

                    removals =[]
                    for i in self.running_tasks:
                        if (self.activities[i].ef <= current_time):
                            self.finished.append(i)
                            self.finished_mark[i ] = 1
                            self.LB[i] = self.activities[i].ef # 修改工序完成下限为实际完成时间
                            self.activities[i].working = False
                            self.activities[i].complete = True
                            self.current_consumption = sub_lists(self.current_consumption,
                                                                 [1 if i == self.activities[activityIndex].resourceRequestH
                                                                 else 0 for i in range(FixedMes.Human_resource_type)])

                            if self.activities[i].RequestStationType >= 0:
                                typeS = self.activities[i].SheiBei[0][0]
                                index = self.activities[i].SheiBei[0][1]
                                self.recordStation[typeS][index] = 0
                                self.Stations[typeS][index].working = False

                            removals.append(i)
                    for i in removals:
                        self.running_tasks.remove(i)

                    if (len(self.finished) == len(self.activities)):
                            break

                    self.eligible = conditionUpdateAndCheck(self.activities,
                                                    self.current_consumption,
                                                    self.finished,
                                                    self.partial_sol_sequeence,
                                                    self.recordStation)

            for num in self.eligible:
                    row = num // self.number_of_opera
                    col = num % self.number_of_opera
                    self.can_be_scheduled_mark[num] = 1

            dfsEF(self.activities, self.number_of_tasks, self.LB)  # 计算每个工序的最早完成时间
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
        self.activities = copy.deepcopy(self.data['activities'])

        #人员设备初始化
        self.init()

        self.t = 0
        for i, activity in self.activities.items():

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

        self.finished_mark = [0 for _ in range(self.number_of_opera * self.number_of_JZJ)]

        #self.finished_time = torch.tensor([0.0 for _ in range(self.number_of_opera * self.number_of_JZJ)])
        # self.scheduled_mark = torch.tensor([[0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)])
        # self.can_be_scheduled_mark = torch.tensor([[0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)])
        self.LB = []

        dfsEF(self.activities,self.number_of_tasks,self.LB) #计算 self.LB
        #self.LB =torch.tensor([0.0 for _ in range(self.number_of_opera * self.number_of_JZJ)])

        self.scheduled_mark = torch.tensor([0.0 for _ in range(self.number_of_opera * self.number_of_JZJ)])
        self.can_be_scheduled_mark = torch.tensor([0.0 for _ in range(self.number_of_opera * self.number_of_JZJ)])

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

        adjacency_matrix = np.eye(self.number_of_tasks, dtype=np.single)
        for node, successors in self.suc:
            for successor in successors:
                for jzj in range(self.number_of_JZJ):
                        adjacency_matrix[jzj * self.number_of_opera + node][
                            jzj * self.number_of_opera + successor] = 1

        self.adj = torch.tensor(adjacency_matrix)

        # 更新紧前节点的紧后节点集合
        for i in range(self.number_of_JZJ * self.number_of_opera):
            if self.activities[i].exist:
                for j in range(self.number_of_JZJ * self.number_of_opera):
                    if self.adj[j][i] == 1:
                        self.adj[j][i] = 0
                        self.adj[j] += self.adj[i]
                        for num in range(len(self.adj[j])):
                            if self.adj[j][num] > 1:
                                self.adj[j][num] = 1
                self.adj[i] = [0 for _ in range(self.number_of_JZJ * self.number_of_opera)]

        fea = torch.stack((self.LB, self.scheduled_mark, self.can_be_scheduled_mark), dim=0)

        self.h_workTime = torch.tensor([0.0 for _ in self.number_of_humans])
        self.h_working = torch.tensor([0 for _ in self.number_of_humans])
        self.can_workTime = torch.tensor([0.0 for _ in self.number_of_humans])

        fea_h = torch.stack((self.h_workTime, self.h_working, self.can_workTime), dim=0)

        #allltasks,current_consumption,running,allNums,finished
        self.eligible = conditionUpdateAndCheck(self.activities,
                                                self.current_consumption,
                                                self.finished,
                                                self.partial_sol_sequeence,
                                                self.recordStation)

        self.mask = np.full(shape=(self.number_of_JZJ * self.number_of_opera), fill_value=0, dtype=bool)


        return fea.to(device),self.adj.to(device), fea_h.to(device)

if __name__ == '__main__':
    env = JZJ(configs.n_j, configs.n_m)
    env.reset()
    for i in range(configs.n_j* configs.n_m):

        env.step(3)
    Draw_gantt(env.Humans)

