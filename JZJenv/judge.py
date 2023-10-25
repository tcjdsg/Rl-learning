import math

from JZJenv.FixedMess import FixedMes


def judgeHuman(Human, type, resourceRequestH1, resourceSumH1, recordH1, now_pos, t, dur):
    # 全甲板模式
    if FixedMes.modeflag == 0:
        typeHuman = Human[type]
    else:
        typeHuman = Human[int((now_pos - 1) / FixedMes.modeflag)][type]

    if resourceRequestH1[type] > 0:
        for human in typeHuman:
            if (len(human.OrderOver) == 0):
                resourceSumH1[type] += 1  # 该类资源可用+1
                recordH1[type].append(human)

            if (len(human.OrderOver) == 1):
                Activity1 = human.OrderOver[0]
                from_pos = Activity1.belong_plane_id
                to_pos = Activity1.belong_plane_id
                movetime1 = 0
                movetime2 = 0
                if (Activity1.ef + round(movetime1, 0)) <= t \
                        or (t + dur) <= (Activity1.es - round(movetime2, 0)):
                    resourceSumH1[type] += 1  # 该类资源可用+1
                    recordH1[type].append(human)

            # 遍历船员工序，找到可能可以插入的位置,如果船员没有工作，人力资源可用
            if (len(human.OrderOver) >= 2):
                flag = False
                for taskIndex in range(len(human.OrderOver) - 1):
                    Activity1 = human.OrderOver[taskIndex]
                    Activity2 = human.OrderOver[taskIndex + 1]

                    from_pos = Activity1.belong_plane_id
                    to_pos = Activity2.belong_plane_id
                    movetime1 = 0
                    movetime2 = 0

                    if (Activity1.ef + round(movetime1, 0)) <= t \
                            and (t + dur) <= (Activity2.es - round(movetime2, 0)):
                        flag = True
                        resourceSumH1[type] += 1  # 该类资源可用+1
                        recordH1[type].append(human)
                        break

                if flag == False:
                    Activity1 = human.OrderOver[0]
                    Activity2 = human.OrderOver[-1]
                    from_pos = Activity2.belong_plane_id
                    to_pos = Activity1.belong_plane_id
                    movetime2 = 0
                    movetime1 = 0

                    if (Activity2.ef + round(movetime2, 0)) <= t \
                            or (t + dur) <= (Activity1.es - round(movetime1, 0)):
                        resourceSumH1[type] += 1  # 该类资源可用+1
                        recordH1[type].append(human)

    return resourceSumH1, recordH1

def judgeStation(now_pos, TaskID,recordStation):

        recordStationID =[]
        # 假设一个工序只需要一个机器
        if FixedMes.OrderInputMes[TaskID][1][1] > 0:
                type = FixedMes.OrderInputMes[TaskID][1][0]
                for stationID in range(len(recordStation[type])):
                # 舰载机在这个加油站的覆盖范围内：
                    if (now_pos+1) in FixedMes.constraintS_JZJ[type][stationID]:
                        # 当前这个站没有工序在进行
                        if recordStation[type][stationID].working == False:
                            recordStationID.append(stationID)
                if len(recordStationID) > 0:
                    flag = True
                else:
                    flag =False

        # 不需要设备
        else:
            flag = True

        return [flag, recordStationID]

#recordStation  可用Station集合 eligibleStation
#recordS  所有Station
def allocationStation(n_orders,activityIndex, recordS, recordStation,dur):

            typeS = FixedMes.OrderInputMes[activityIndex%n_orders][1][0]
            need =  FixedMes.OrderInputMes[activityIndex%n_orders][1][1]
            if need > 0:
                alreadyWorkTime = math.inf
                index = 0
                for stationID in recordStation:
                        nowStaion = recordS[typeS][stationID]
                        if nowStaion.alreadyworkTime < alreadyWorkTime:
                            alreadyWorkTime = nowStaion.alreadyworkTime
                            index = nowStaion.zunumber
                # 更新
                recordS[typeS][index].update(activityIndex,dur)
                recordS[typeS][index].working = True
                return [typeS,index]
            return []
def allocationStationStatic(n_orders,activityIndex, recordS, dur):
    typeS = FixedMes.OrderInputMes[activityIndex % n_orders][1][0]
    need = FixedMes.OrderInputMes[activityIndex % n_orders][1][1]
    if need > 0:
        alreadyWorkTime = math.inf
        index = 0
        for stationID in range(recordS[typeS]):
            nowStaion = recordS[typeS][stationID]
            if nowStaion.alreadyworkTime < alreadyWorkTime:
                alreadyWorkTime = nowStaion.alreadyworkTime
                index = nowStaion.zunumber
        # 更新
        recordS[typeS][index].update(activityIndex, dur)
        recordS[typeS][index].working = True
        return [typeS, index]
    return []
def allocationHuman(task, Humans):

    for type in range(len(task.resourceRequestH)):
        record = []
        if task.resourceRequestH[type]>0:
            eligibleHuman = Humans[type]
            for human in eligibleHuman:
                if human.working == False:
                    record.append(human)

            record = sorted(record,key=lambda x:(x.alreadyworkTime,
                                                 FixedMes.distance[x.OrderOver[-1].belong_plane_id][task.belong_plane_id] if len(x.OrderOver)>0
                                                 else FixedMes.distance[0][task.belong_plane_id]))
            need = task.resourceRequestH[type]

            while need>0:

                Humans[type][record[0].zunumber].update(task)
                Humans[type][record[0].zunumber].working = True
                task.HumanNums.append([type, record[0].zunumber])
                record.remove(record[0])
                need-=1


