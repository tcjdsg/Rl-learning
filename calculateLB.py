import networkx as nx
from JZJenv.FixedMess import FixedMes
from Params import configs

def newAON(Humanss,Stations,Spacess,suc,time):
    edge = []
    for from_, to_list in suc.items():
        for toid in to_list:
            edge.append((from_, toid, time[from_]))
    # print("原始边",edge)
    human_moveTime = []
    for humans in Humanss:
        for human in humans:
            movetime = add(edge, human.OrderOver, time, 'human')
            human_moveTime.append(movetime)
    for stations in Stations:
        for station in stations:
            add(edge, station.OrderOver, time, 'station')

    for space in Spacess:
            add(edge, space.OrderOver, time, 'space')
    # 这里面就包含了新的约束，但是燃气等资源约束暂时还未考虑
    return edge,human_moveTime

def add(edge, OrderOver, time, workType):
    Order = OrderOver
    move = 0
    for activity in range(len(Order)):
        frontActivity = Order[activity]
        if activity < len(Order) - 1:  # 说明这个人后面还有任务要干
                backActivity = Order[activity + 1]
                frontId = frontActivity // configs.n_jzjs
                backId = backActivity // configs.n_jzjs
                if workType == 'human':
                    edge.append((frontId, backId, time[frontActivity] + FixedMes.distance[frontId][backId] * FixedMes.human_walk_speed))
                    move += FixedMes.distance[frontId][backId] * FixedMes.human_walk_speed
                if workType == 'station':
                    edge.append((frontId, backId, time[frontActivity] + FixedMes.distance[frontId][backId] / FixedMes.station_tranfer_speed))
                if workType == 'space':
                    edge.append((frontId, backId, time[frontActivity]))
    return move

#求关键路径
def CPM(newedge):

    DG = nx.DiGraph()  # 创建：空的 有向图
    DG.add_nodes_from(range(0, configs.n_jzjs * configs.n_orders), VE=0, VL=0)
    DG.add_weighted_edges_from(newedge)

    lenNodes = len(DG.nodes)  # 顶点数量 YouCans
    topoSeq = list(nx.topological_sort(DG))  # 拓扑序列: [1, 3, 4, 2, 5, 7, 6, 8]

    lenCriticalPath = nx.dag_longest_path_length(DG)  # 关键路径的长度
    return lenCriticalPath  # 51

def calculateLBs(Humanss, Stations, Spacess, suc, time):
    newedge,move_time = newAON(Humanss, Stations, Spacess, suc, time)
    graph = {}
    for a, b, t in newedge:
        if b not in graph:
            graph[b] = {}
        if a not in graph:
            graph[a] = {}
        graph[b][a] = t
    LB = [0 for _ in range(len(graph.keys()))]
    dfsEF(graph, len(graph.keys()) - 1, LB)

    can_workTime = cal_Humans_endTime(Humanss, LB)
    return LB, can_workTime, move_time

def dfsEF(pre, i, LB):
    if len(pre[i].keys()) == 0:
        LB[i] = 0
        return 0
    time = 0
    for Orderid in pre[i]:
        time = max(time, dfsEF(pre, Orderid, LB) + pre[i][Orderid])
    LB[i] = time
    return LB[i]

def cal_Humans_endTime(Humanss , LB):
    times = [[0.0 for _ in range(len(Humanss[i]))] for i in range(len(Humanss))]
    time = []
    for i in range(len(Humanss)):
        for j in range(len(Humanss[i])):
            if len(Humanss[i][j].OrderOver)==0:
                times[i][j] = 0
            else:
                acindex = Humanss[i][j].OrderOver[-1]
                times[i][j] = LB[acindex]
            time.append(times[i][j])
    return time

def condition_check( pre, exist, finished_mark, partitial):
    precedence_eligible = []# 满足紧前工序已完成的工序

    for i in range(len(finished_mark)):
        if i in partitial or finished_mark[i]==1 or exist[i]==0:
            continue
        flag = True
        prenumber = pre[i]# 前序
        for ordernumber in prenumber:
            if finished_mark[ordernumber] == 0:
                flag = False
                break
        if flag == True:
            precedence_eligible.append(i)
    # for i in precedence_eligible:
    #     row = i // n_orders
    #     col = i % n_orders
    #     consump = [FixedMes.OrderInputMes[col][0][1] if i == FixedMes.OrderInputMes[col][0][0] else 0 for i in
    #                 range(configs.Human_resource_type)]
    #
    #     if (less_than(consump, sub_lists(configs.total_Human_resource, current_consumption))) and judgeStation(row, col,Stations)[0]:
    #         eligible.append(i)
    return precedence_eligible

if __name__ == '__main__':
    edge=[]
    edge.append((0,1,1.5))
    edge.append((0,5, 2.5))
    edge.append((1,3,4))
    edge.append((1,7,5))
    edge.append((2,1,1))
    edge.append((4,5,2))
    edge.append((5,7,2))
    edge.append((6,5,2))
    edge.append((3, 8, 0))
    edge.append((7, 8, 0))

    distances = calculateLBs(edge)
    print(distances)

# def serial_sgs( option='forward',n_tasks=0, priority_rule='LFT', priorities=[], stat='min'):
#         """
#             Implements the Serial Schedule Generation Scheme
#
#             Parameters:
#                 option : Forward or reverse scheduling
#                 priority _rule : Priority rule used. One of ['EST','EFT','LST','LFT','SPT','FIFO','MTS','RAND','GRPW','GRD']
#
#             Returns:
#                 Tuple of (Fractional deviation , makespan)
#                 Fractional deviation = (makespan-self.mpm_time)/self.mpm_time,makespan
#         """
#         # if priority_rule in ['GRPW', 'GRD']:
#         #     calculate_grpw()
#         #     calculate_grd()
#
#         # Initialize arrays to store computed values
#         start_times = [0] * (n_tasks)  # Start times of schedule
#         finish_times = [0] * (n_tasks)  # Finish times of schedule
#         earliest_start = [0] * (n_tasks)  # Earliest precedence feasible start times(Different from EST)
#         resource_consumption = [[0 for col in range(configs.Human_resource_type)] for row in range(all_time + 1)]  # 2D array of resource consumption of size n x k
#         scheduled = [0] * (n_tasks + 1)  # Boolean array to indicate if job is scheduled
#         if (option == 'forward'):
#             #graph = self.G  # If forward scheduling use graph as it is
#             start_vertex = 1
#             predecessors = pre
#         else:  # option = reverse
#             #graph = self.G_T  # If reverse scheduling use transpose of grapj
#             start_vertex = n_tasks
#             predecessors = suc
#         start_times[start_vertex] = 0  # Schedule the first dummy job
#         finish_times[start_vertex] = 0
#         scheduled[start_vertex] = 1
#         for g in range(0, n_tasks):  # Perform n-1 iterations (Dummy job already scheduled)
#             eligible = []  # List of eligible jobs based on precedence only
#
#             # For each unscheduled job check if it is eligible
#             for i in range(1, self.n_jobs + 1):
#                 if (scheduled[i] == 0):
#                     con = True
#                     for j in predecessors[i]:
#                         if scheduled[j] == 0:
#                             con = False
#                             break
#                     if (con):
#                         eligible.append(i)
#             choice = self.choose(eligible, priority_rule=priority_rule,
#                                  priorities=priorities)  # Choose a job according to some priority rule
#             max_pred_finish_time = 0  # Find the maximum precedence feasible start time for chosen job
#             for i in predecessors[choice]:
#                 max_pred_finish_time = max(finish_times[i], max_pred_finish_time)
#             earliest_start[choice] = max_pred_finish_time + 1  # Update the found value in array
#             scheduled[choice] = 1
#             feasible_start_time = self.time_resource_available(choice, earliest_start[
#                 choice])  # Find the earliest resource feasible time
#             start_times[choice] = feasible_start_time
#             finish_times[choice] = feasible_start_time + self.durations[choice] - 1  # Update finish time
#             for i in range(feasible_start_time, finish_times[choice] + 1):
#                 self.resource_consumption[i] = add_lists(self.resource_consumption[i],
#                                                          self.job_resources[choice])  # Update resource consumption
#         makespan = max(finish_times)  # Makespan is the max value of finish time over all jobs
#         if (option != 'forward'):
#             for i in range(1, self.n_jobs + 1):
#                 finish_times[i] = makespan - start_times[i]
#                 start_times[i] = finish_times[i] - self.durations[i] + 1
#         self.serial_finish_times = finish_times
#         return (makespan - self.mpm_time) / self.mpm_time, makespan


# def get_order_human_normal(need_guarantee, eligible,finished_mark,running_tasks,human_state, recordWorkHumans):
#     iter = Chromosome()
#     moment = 0
#     move_time_human = 0
#     order_number = 0
#     min_time = float('inf')
#     old_min_time = float('inf')
#     all_plane_order = configs.n_orders * configs.n_jzjs
#     human_ready = [ False for _ in range(all_plane_order)]
#     working = [ False for _ in range(all_plane_order)]
#
#
#     while order_number != all_plane_order:
#         for typeH in range(len(human_state)):
#             for index in range(len(human_state[typeH])):
#                 if len(human_state[typeH][index].wait_fini_order) > 0:
#                     order_first = human_state[typeH][index].wait_fini_order[0]
#                     jzj = order_first // configs.n_orders
#                     number_task = order_first % configs.n_orders
#                     if  human_state[typeH][index].working == False:
#                         if order_first in eligible and not finished_mark[order_first] and not running_tasks[order_first]:
#                             sign = True
#                             for sign1 in recordWorkHumans[order_first]:
#                                 typeHH = sign1[0]
#                                 assert typeHH == typeH
#                                 indexHH = sign1[1]
#                                 needH = FixedMes.OrderInputMes[number_task][0][1]
#                                 if True:
#                                     if not human_state[typeHH][indexHH].working:
#                                         sign = True
#                                     else:
#                                         sign = False
#                                         break
#                             if sign: #任务要求的人员都处于空闲中
#                                 human_ready[order_first] = True #可以开始工作了
#                                 working[order_first] = True
#                                 # 就在这里加行走时间吧
#                                 max_time_walk = 0
#                                 for sign1 in recordWorkHumans[order_first]:
#                                     typeHH = sign1[0]
#                                     assert typeHH == typeH
#                                     indexHH = sign1[1]
#                                     if human_state[typeHH][indexHH].OrderOver: #TODO orderover和wait_order 区分一下
#                                         continue # 因为是第一个 不需要行走
#                                     temporary_walk = HumanWork()
#                                     size = len(human_state[typeHH][indexHH].OrderOver)
#                                     temporary_walk.from_pid = human_state[typeHH][indexHH].OrderOver[size - 1]//configs.n_orders #TODO 感觉还是得搞个plane_id
#                                     temporary_walk.to_pid = jzj
#                                     temporary_walk.walk_distance = FixedMes.distance[temporary_walk.from_pid][
#                                         temporary_walk.to_pid]
#                                     temporary_walk.walk_start_time = moment
#                                     temporary_walk.walk_time = int(temporary_walk.walk_distance / temporary_walk.walk_speed)
#                                     if temporary_walk.walk_distance != 0:
#                                         human_state[typeHH][indexHH].walk_task.append(temporary_walk)
#                                     max_time_walk = max(max_time_walk, temporary_walk.walk_time)
#                                 order_first.time_start = moment + max_time_walk
#                                 for sign1 in order_first.humannum:
#                                     human_state[sign1 - 1].now_order = order_first.order_num
#                                     human_state[sign1 - 1].state = True
#                                     human_state[sign1 - 1].wait_fini_order[0] = order_first.myclone()
#                                 need_guarantee[order_first.belong_plane_id - 1][
#                                     order_first.order_num - 1] = order_first.myclone()
#
#                     if human_state[human_alone].state:
#                         if order_first.time_start + order_first.time_during <= moment:
#                             order_first.time_end = moment
#                             order_first.working = False
#                             order_first.complete = True
#                             for j in order_first.humannum:
#                                 human_state[j - 1].now_order = 0
#                                 human_state[j - 1].time_during += order_first.time_during
#                                 human_state[j - 1].state = False
#                                 human_state[j - 1].order_over.append(order_first.myclone())
#                                 human_state[j - 1].wait_fini_order.pop(0)
#                             order_number += 1
#                             need_guarantee[order_first.belong_plane_id - 1][
#                                 order_first.order_num - 1] = order_first.myclone()
#                             human_alone -= 1
#                         else:
#                             min_time = min(min_time, order_first.time_start + order_first.time_during)
#
#             if check_condition(need_guarantee):
#                 continue
#
#             if min_time != float('inf'):
#                 old_min_time = moment
#                 moment = min_time
#             min_time = float('inf')
#
#             if min_work_time <= moment:
#                 return None
#
#             for i in human_state:
#                 for time_walk in i.walk_task:
#                     move_time_human += time_walk.walk_time
#
#             work_time = [i.time_during for i in human_state]
#             mean = sum(work_time) / len(work_time)
#             accum = sum([(x - mean) ** 2 for x in work_time])
#             stdev = (accum / len(work_time)) ** 0.5
#
#             iter.movetime = move_time_human
#             iter.variance = stdev
#             iter.worktime = moment
#
#         return iter

# def check_condition(need_guarantee):
#     work_sign = False
#     for i in range(len(need_guarantee)):
#         for j in range(plane_order_num):
#             if need_guarantee[i][j].complete:
#                 continue
#             sign = True
#             for in_val in need_guarantee[i][j].pre_order:
#                 if not need_guarantee[i][in_val - 1].complete:
#                     sign = False
#                     break
#             if sign and not need_guarantee[i][j].condition:
#                 need_guarantee[i][j].condition = True
#                 work_sign = True
#     return work_sign
