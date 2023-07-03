import copy

from calPriority import *
from utils import *
import gym
import random
import numpy as np
from Params import configs
from agent_utils import  conditionUpdateAndCheck
from  read.preprocess import InitM
from agent_utils import override

from JZJenv.Activitity import Order
from JZJenv.FixedMess import FixedMes
from JZJenv.Human import Human
from JZJenv.Station import Station

filenameDis = "dis.csv"
class JZJ(gym.Env):
    def __init__(self,
                 planeNum,
                 planeOrderNum):


        self.step_count = 0
        self.number_of_JZJ = planeNum
        self.number_of_opera = planeOrderNum
        self.number_of_tasks = self.number_of_JZJ * self.number_of_opera
        self.Init = InitM(filenameDis)
        FixedMes.distance = self.Init.readDis()
        self.static_activities, self.suc ,self.pre= self.Init.readData()
        calLFTandMTS(self.static_activities)

        # the task id for first column
        # self.first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
        # # the task id for last column
        # self.last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]
        self.finished_time = [[0.0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)]
        self.scheduled_mark = [[0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)]
        self.can_be_scheduled_mark = [[0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)]
        self.running_tasks = []
        self.waiting_tasks = []
        self.scheduled_tasksID = []


    def done(self):
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            return True
        return False

    @override
    def step(self, action):
        pri = ['LFT', 'LST', 'EST', 'EFT', 'FIFO', 'RAND', 'SPT', 'MTS', 'GRPW', 'GRD', 'IRSM', 'WCS', 'ACS']
        priority_rule = pri[action]
        current_time = self.t
        removals=[]

        if len(self.eligible) > 1:
            #alltasks,eligible, current_time, current_consumption, active_list, finish_times
            calculate_dynamic_priority_rules(self.activities,
                                             self.eligible,
                                             current_time,
                                             self.current_consumption,
                                             self.running_tasks,
                                             self.finished_time)

        #基于规则选择工件
        activityIndex = self.returnActivity(priority_rule)

        self.can_be_scheduled_mark = [[0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)]


        current_time = self.t
        maxend_Time = self.Cmax

        if activityIndex is not None and activityIndex not in self.partial_sol_sequeence:

            # 更新信息
            self.eligible.remove(activityIndex)
            row = activityIndex // self.number_of_JZJ
            col = activityIndex % self.number_of_JZJ
            self.step_count += 1
            self.partial_sol_sequeence.append(activityIndex)

            # 更新状态，状态包括：
            self.scheduled_mark[row][col] = 1
            self.activities[activityIndex].scheduled = True
            self.activities[activityIndex].working = True

            dur_a = getTime(row, col)


            self.activities[activityIndex].es = current_time
            self.activities[activityIndex].ef = current_time + dur_a
            self.current_consumption = add_lists(self.current_consumption,
                                                 self.activities[activityIndex].resourceRequestH)

            self.running_tasks.append(activityIndex)
            self.end_Time.append(current_time + dur_a)
            maxend_Time = sorted(self.end_Time, key=lambda x:x)[-1]


            if len(self.eligible) > 0:
                #如果还有可用的，就还是这个集合
                self.t = current_time

            else:
                #找到新的最早结束时间
                self.t = sorted([self.activities[i].ef for i in self.running_tasks])[0]
                current_time = self.t
                for i in self.running_tasks:
                    if (self.activities[i].ef <= current_time):
                        self.finished.append(i)
                        self.finished_mark[i // self.number_of_JZJ][i % self.number_of_JZJ] = 1
                        self.finished_time[i // self.number_of_JZJ][i % self.number_of_JZJ] = self.activities[i].ef
                        self.activities[i].working = False
                        self.activities[i].complete = True
                        self.current_consumption = sub_lists(self.current_consumption,
                                                             self.activities[i].resourceRequestH)
                        removals.append(i)

                for i in removals:
                    self.running_tasks.remove(i)

                self.eligible = conditionUpdateAndCheck(self.activities,
                                                    self.current_consumption,
                                                    self.finished)

            for num in self.eligible:
                    row = num // self.number_of_JZJ
                    col = num % self.number_of_JZJ
                    self.can_be_scheduled_mark[row][col] = 1

        # prepare for return
        # 3 * JZJNumber * JZJNumber

        fea =  np.array([self.finished_time/np.max(self.finished_time),self.scheduled_mark,self.can_be_scheduled_mark])
        reward = - (maxend_Time - self.Cmax)
        if reward == 0:
            reward = configs.rewardscale
            self.posRewards += reward
        self.Cmax = maxend_Time

        return fea, reward, self.done(), self.Cmax

    def returnActivity(self,priority_rule):
        value_lists=[]

        if (priority_rule == 'LFT'):
            for i in self.eligible:
                value_lists.append(self.activities[i].lf)
            return self.eligible[find_index(self.eligible, value_lists, 'min')]
        elif (priority_rule == 'LST'):
            for i in self.eligible:
                value_lists.append(self.activities[i].ls)
            return self.eligible[find_index(self.eligible, value_lists, 'min')]
        elif (priority_rule == 'EST'):
            for i in self.eligible:
                value_lists.append(self.activities[i].es)
            return self.eligible[find_index(self.eligible, value_lists, 'min')]
        elif (priority_rule == 'EFT'):
            for i in self.eligible:
                value_lists.append(self.activities[i].ef)
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


    @override
    def reset(self):
        self.t=0
        for i,activity in self.static_activities.items():
            activity.es = 0
            activity.ef = 0
            activity.ls = 0
            activity.lf = 0
            activity.tf = 0
            activity.mts =0
            activity.grpw = 0
            activity.GRD=0

            activity.acs=0
            activity.wsc=0
            activity.irsm=0

            activity.scheduled = False
            activity.complete = False

            activity.HumanNums = []  # 执行任务的人员编号
            activity.SheiBei = []

        self.activities = self.static_activities
        self.finished_mark = [[0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)]

        self.finished_time = [[0.0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)]
        self.scheduled_mark = [[0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)]
        self.can_be_scheduled_mark = [[0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)]

        self.running_tasksID = []
        self.end_Time = []
        self.finished =[]
        self.waiting_tasksID = []

        self.step_count = 0

        # record action history
        self.partial_sol_sequeence = []
        self.flags = []
        self.posRewards = 0
        self.initQuality=0
        self.Cmax = 0

        self.current_consumption = [0 for _ in range(FixedMes.total_Huamn_resource)]
        fea =  np.array([self.finished_time,self.scheduled_mark,self.can_be_scheduled_mark])
        #allltasks,current_consumption,running,allNums,finished
        self.eligible = conditionUpdateAndCheck(self.static_activities,self.current_consumption, self.finished)


        return  fea
