import copy
from utils import *
import gym
import numpy as np
from Params import configs
from agent_utils import conditionCheck
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
        self.can_be_scheduled_mark = [[0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)]
        #  action表示选择的活动[jzjnumber,operationNumber]

        # action is a int 0 - jzjnumber * operationNumber
        # redundant action makes no effect
        running =[]
        startTime_a = FixedMes.t
        eligible=[]
        if action not in self.partial_sol_sequeence:

            # 更新信息
            row = action // self.number_of_JZJ
            col = action % self.number_of_JZJ
            self.step_count += 1
            self.partial_sol_sequeence.append(action)

            # 更新状态，状态包括：
            self.scheduled_mark[row][col] = 1
            dur_a = getTime(row, col)
            self.activities[action].dur = dur_a
            self.finished_time[row][col] = startTime_a + dur_a

            self.activities[action].scheduled = True
            self.activities[action].working = True
            self.activities[action].es  = startTime_a
            self.activities[action].ef  = startTime_a + dur_a
            self.running_tasks.append(self.activities[action])
            running = sorted(self.running_tasks, key=lambda x:x.ef)
            # self.waiting_tasks.remove(self.activities[action])

            eligible = conditionCheck(self.activities, self.pre, FixedMes.Activity_num, self.partial_sol_sequeence)
            for num in eligible:
                row = num // self.number_of_JZJ
                col = num % self.number_of_JZJ
                self.can_be_scheduled_mark[row][col] = 1
            # permissible left shift
            # startTime_a, flag = permissibleLeftShift(a=action, durMat=self.dur, mchMat=self.m, mchsStartTimes=self.mchsStartTimes, opIDsOnMchs=self.opIDsOnMchs)
            # self.flags.append(flag)
            # # update omega or mask
            # if action not in self.last_col:
            #     self.omega[action // self.number_of_machines] += 1
            # else:
            #     self.mask[action // self.number_of_machines] = 1
            #
            # self.temp1[row, col] = startTime_a + dur_a

            # adj matrix
            # precd, succd = self.getNghbs(action, self.opIDsOnMchs)
            # self.adj[action] = 0
            # self.adj[action, action] = 1
            # if action not in self.first_col:
            #     self.adj[action, action - 1] = 1
            # self.adj[action, precd] = 1
            # self.adj[succd, action] = 1
            # if flag and precd != action and succd != action:  # Remove the old arc when a new operation inserts between two operations
            #     self.adj[succd, precd] = 0

        # prepare for return
        # 3 * JZJNumber * JZJNumber
        fea =  np.array([self.finished_time,self.scheduled_mark,self.can_be_scheduled_mark])
        reward = - (running[-1].ef - self.Cmax)
        if reward == 0:
            reward = configs.rewardscale
            self.posRewards += reward
        self.Cmax = running[-1].ef
        return fea, reward, self.done(), eligible

    @override
    def reset(self, data):
        self.activities = copy.deepcopy(self.static_activities)

        self.finished_time = [[0.0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)]
        self.scheduled_mark = [[0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)]
        self.can_be_scheduled_mark = [[0 for _ in range(self.number_of_opera)] for _ in range(self.number_of_JZJ)]
        self.running_tasks = []
        self.waiting_tasks = []
        self.scheduled_tasksID = []
        self.partial_sol_sequeence=[]
        self.step_count = 0
        self.m = data[-1]
        self.dur = data[0].astype(np.single)
        self.dur_cp = np.copy(self.dur)
        # record action history
        self.partial_sol_sequeence = []
        self.flags = []
        self.posRewards = 0
        self.initQuality=0
        self.Cmax = 0

        self.finished_mark = np.zeros_like(self.m, dtype=np.single)

        fea =  np.array([self.finished_time,self.scheduled_mark,self.can_be_scheduled_mark])
        eligible = conditionCheck(self.activities, self.pre, FixedMes.Activity_num, self.partial_sol_sequeence)

        return  fea ,eligible
