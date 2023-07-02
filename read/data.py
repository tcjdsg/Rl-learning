#读取数据
# -*- coding: utf-8 -*-
import copy
from collections import defaultdict

from conM.FixedMess import FixedMes
from activity.Activitity import Order
from conM import FixedMess

def readData():
    '''
    包括  1.活动数   2.项目资源数 3.项目资源种类数   4.项目资源限量
    5.所有活动的ID，持续时间，资源需求，紧前活动
    :param fileName:
    :return: activities:这个是标准单机流程
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
    for i in range(1, FixedMes.Activity_num):
        preActDict[i].append(0)
    index = 0

    # 构建任务网络
    for i in range(FixedMes.planeNum):
        jzjNumber = FixedMes.jzjNumbers[i]
        for j in range(FixedMes.planeOrderNum):
            index += 1
            taskId = j + 1
            duration = FixedMes.OrderTime[taskId]
            resourceH = [0 for _ in range(FixedMes.Human_resource_type)]
            # nt()pri
            resourceH[FixedMes.OrderInputMes[taskId][0][0]] = FixedMes.OrderInputMes[taskId][0][1]

            resourceS = [0 for _ in range(FixedMes.station_resource_type)]
            resourceS[FixedMes.OrderInputMes[taskId][1][0]] = FixedMes.OrderInputMes[taskId][1][1]

            resourceSpace = [0 for _ in range(FixedMes.space_resource_type)]
            if FixedMes.OrderInputMes[taskId][2][1] > 0:
                resourceSpace[jzjNumber - 1] = FixedMes.OrderInputMes[taskId][2][1]

            SUCOrder = [num + (FixedMes.planeOrderNum) * i for num in FixedMes.SUCOrder[taskId]]

            if len(FixedMes.SUCOrder[taskId]) == 0:
                SUCOrder.append((FixedMes.planeOrderNum) * FixedMes.planeNum + 1)

            task = Order(index, taskId, duration, resourceH, resourceS, resourceSpace, SUCOrder, jzjNumber)

            activities[index] = task
            for s in SUCOrder:
                preActDict[s].append(index)

    for act in activities.keys():
        activities[act].predecessor = preActDict[act]

    SUCOrder = [i * FixedMes.planeOrderNum + 1 for i in range(0, FixedMes.planeNum)]
    resourceH = [0 for _ in range(FixedMes.Human_resource_type)]
    resourceS = [0 for _ in range(FixedMes.station_resource_type)]
    resourceSpace = [0 for _ in range(FixedMes.space_resource_type)]

    activities[0] = Order(0, 0, 0, resourceH, resourceS, resourceSpace, SUCOrder, 0)
    activities[0].predecessor = []

    activities[index + 1] = Order(index + 1, 0, 0, resourceH, resourceS, resourceSpace, [], 0)

    activities[index + 1].predecessor = [i * FixedMes.planeOrderNum for i in range(1, FixedMes.planeNum + 1)]

    return activities
    # 活动数int， 资源数int， 资源限量np.array， 所有活动集合dic{活动代号：活动对象}