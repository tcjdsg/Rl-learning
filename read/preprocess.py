#读取数据
# -*- coding: utf-8 -*-
import copy
from collections import defaultdict

import numpy as np
import pandas as pd
from JZJenv.FixedMess import FixedMes
from JZJenv.Activitity import Order


class InitM(object):
    def __init__(self, filenameDis):

        self.filename1 = filenameDis
        self.humanMoveTime = []
        self.num_activities = 0
        self.num_resource_type = 0
        self.total_resource = []
        self.activities = { }
    def readDis(self):

        dis = pd.read_csv(self.filename1, header=None,encoding="utf-8").values
        pdis = dis.tolist()
        for i in range(1,dis.shape[0]):
            for j in range(1,dis.shape[1]):
                pdis[i][j] = round(dis[i][j] * 1.0  , 1 )#单位是m
                #不考虑设备接口转移速度
        return pdis

    def readData(self):

        '''
        包括  1.活动数   2.项目资源数 3.项目资源种类数   4.项目资源限量
        5.所有活动的ID，持续时间，资源需求，紧前活动
        :param fileName:
        :return: activities:这个是标准单机流程


        搞个随机的时间和工序数量
        '''


        # f = open(self.filename2)
        # jzjnums = f.readline().split(' ')
        # jzjNumbers = [ int(jzjnums[i]) for i in range(len(jzjnums))]
        # taskAndResourceType = f.readline().split(' ')  # 第一行数据包含活动数和资源数
        # num_activities = int(taskAndResourceType[0])  # 得到活动数
        # num_resource_type = int(taskAndResourceType[1])  # 得到资源类数
        # total_resource = np.array([int(value) for value in f.readline().split(' ')[:]])  # 获取资源限量
        # 将每个活动的所有信息存入到对应的Activity对象中去
        activities = {}

        preActDict = defaultdict(lambda: [])
        sucActDict = defaultdict(lambda: [])

        index = 0

        # 构建任务网络
        for i in range(FixedMes.planeNum):
            jzjNumber = FixedMes.jzjNumbers[i]
            for j in range(FixedMes.planeOrderNum):

                taskId = j
                duration = FixedMes.OrderTime[taskId]
                vacp = FixedMes.VACP[taskId]
                resourceH = [0 for _ in range(FixedMes.Human_resource_type)]
                # nt()pri
                resourceH[FixedMes.OrderInputMes[taskId][0][0]] = FixedMes.OrderInputMes[taskId][0][1]

                resourceS = -1
                if FixedMes.OrderInputMes[taskId][1][1]>0:
                    resourceS = FixedMes.OrderInputMes[taskId][1][0]
                resourceSpace = [0 for _ in range(FixedMes.space_resource_type)]
                if FixedMes.OrderInputMes[taskId][2][1]>0:
                    resourceSpace[jzjNumber-1] = FixedMes.OrderInputMes[taskId][2][1]

                SUCOrder = [num+(FixedMes.planeOrderNum)*i for num in FixedMes.SUCOrder[taskId]]

                sucActDict[index] = SUCOrder

                task = Order(index, taskId,duration, resourceH,resourceS,resourceSpace,SUCOrder,jzjNumber)
                task.vacp = vacp
                activities[index] = task

                for s in SUCOrder:
                    preActDict[s].append(index)

                index += 1

        for act in activities.keys():
            activities[act].predecessor = preActDict[act]

        return  activities,sucActDict,preActDict
        # 活动数int， 资源数int， 资源限量np.array， 所有活动集合dic{活动代号：活动对象}


if __name__ == '__main__':
    m = InitM("dis.csv")
    m.readDis()
    m.readData()
    print()












