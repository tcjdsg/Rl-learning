from collections import defaultdict

class FixedMes(object):
    """
    distance:
    orderInputMes:
    """
    pri = ['LFT','LST','FIFO','RAND','SPT','MTS','GRPW','GRD','IRSM','WCS','ACS']
    #全局时钟
    t = 0

    distance = [[0,10,20,30,40,53,62,73,84,126,139,147],
                [10,0,10,20,30,43,52,63,74,117,130,138],
                [20,10,0,10,20,33,42,53,64,108,121,129],
                [30,20,10,0,10,23,32,43,54,101,114,122],
                [5,40,30,20,10,0,13,22,33,44,97,110,117],
                [53,43,33,23,13,0,9,20,31,89,101,108],
                [62,52,42,32,22,9,0,11,22,82,94,100],
                [73,63,53,43,33,20,11,0,12,77,89,95],
                [84,74,64,54,44,31,22,12,0,78,88,93],
                [126,117,108,101,97,89,82,77,78,0,13,21],
                [139,130,121,114,110,101,94,89,88,13,0,9],
                [147,138,129,122,117,108,100,95,93,21,9,0]]

    #座舱限制。相当于是每个站位都有一个座舱，每个舰载机只能用自己座舱。
    # space_resource_type = planeNum
    # total_space_resource = [1 for i in range(planeNum)]

    # total_Huamn_resource = [30]
    constraintOrder = defaultdict(lambda: []) #记录每类人的可作用工序，和可作用舰载机范围
    # constraintOrder[0] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    constraintOrder[0] = [1, 2, 5]
    constraintOrder[1] = [3, 4, 16,17]
    constraintOrder[2] = [7, 8, 14,16]
    constraintOrder[3] = [6, 9, 10, 11,12,13,15]

    modeflag = 0 #0是单机、1是全甲板，这里考虑全甲板，如果是全甲板
    # constraintJZJ = defaultdict(lambda: []) #保障人员可作用舰载机范围,两种模式，单机或者全甲板

    station_resource_type = 5
    #加油、供电、充氧、充氮、液压
    total_station_resource = [6,12,5,5,6]

    #飞机数量比较少的时候，这些燃料资源的限制约束不起作用。
    # total_renew_resource = [5,5,2,4,2]
    # # total_renew_resource = [1,1,1,1,1]
    total_renew_resource = [99,99,99,99,99]

    constraintS_Order = defaultdict(lambda: [])  # 记录每类设备的可作用工序，和可作用舰载机范围

     # 设备保障范围约束
    constraintS_JZJ = defaultdict(lambda: [])

    constraintS_JZJ[0] = [[1, 2, 3],
                          [3, 4, 5],
                          [6, 7],
                          [7,8,9],
                          [9,10,11],
                          [12]]

    constraintS_JZJ[1] = [[1],
                          [2],
                          [3],
                          [4],
                          [5],
                          [6],
                          [7],
                          [8],
                          [9],
                          [10],
                          [11],
                          [12]
                          ]
    constraintS_JZJ[2] = [[1,2,3,4],
                          [4,5,6,7],
                          [7,8,9],
                          [9,10,11],
                          [12]
                          ]

    constraintS_JZJ[3] = [[1,2,3,4],
                          [4,5,6,7],
                          [7,8,9],
                          [9,10,11],
                          [12]
                          ]

    constraintS_JZJ[4] = [[1,2,3],
                          [2,3,4],
                          [4,5,6],
                          [7,8,9],
                          [9,10,11],
                          [12]
                          ]


    #工序顺序
    SUCOrder = defaultdict(lambda: [])
    SUCOrder[0] = [1, 3, 5, 8, 9, 10, 11, 12, 13]
    SUCOrder[1] = [2]
    SUCOrder[2] = [14]
    SUCOrder[3] = [4]
    SUCOrder[4] = [14]
    SUCOrder[5] = [6]
    SUCOrder[6] = [7]
    SUCOrder[7] = [14]
    SUCOrder[8] = [14]
    SUCOrder[9] = [14]
    SUCOrder[10] = [17]
    SUCOrder[11] = [14]
    SUCOrder[12] = [17]
    SUCOrder[13] = [17]
    SUCOrder[14] = [15, 16]
    SUCOrder[15] = [17]
    SUCOrder[16] = [17]
    SUCOrder[17] = [18]
    SUCOrder[18] = []

    #工序顺序
    PREOrder = defaultdict(lambda: [])
    PREOrder[0] = []
    PREOrder[1] = [0]
    PREOrder[2] = [1]
    PREOrder[3] = [0]
    PREOrder[4] = [3]
    PREOrder[5] = [0]
    PREOrder[6] = [5]
    PREOrder[7] = [6]
    PREOrder[8] = [0]
    PREOrder[9] = [0]
    PREOrder[10] = [0]
    PREOrder[11] = [0]
    PREOrder[12] = [0]
    PREOrder[13] = [0]
    PREOrder[14] = [2, 4, 7, 8, 9, 11, 12]
    PREOrder[15] = [14]
    PREOrder[16] = [14]
    PREOrder[17] = [15, 16, 10, 12, 13]
    PREOrder[18] = [17]

    #特设 航电 机械 军械
    OrderInputMes = [
                     [(0, 0), (0, 0), (0, 0)],  # 虚拟1
                     [(0, 1), (0, 0), (0, 0)],  # 2
                     [(0, 1), (1, 1), (0, 1)],  # 3
                     [(1, 1), (0, 0), (0, 0)],  # 4
                     [(1, 1), (1, 1), (0, 1)],  # 5
                     [(3, 1), (2, 1), (0, 0)],  # 6
                     [(3, 2), (0, 1), (0, 0)],  # 7
                     [(3, 2), (0, 0), (0, 0)],  # 8,
                     [(2, 1), (1, 1), (0, 1)],  # 9
                     [(2, 1), (0, 0), (0, 1)],  # 10
                     [(0, 1), (3, 1), (0, 0)],  # 11
                     [(2, 2), (0, 0), (0, 0)],  # 12
                     [(2, 1), (0, 0), (0, 0)],  # 13
                     [(2, 1), (4, 1), (0, 0)],  # 14
                     [(0, 1), (0, 0), (0, 0)],  # 15
                     [(3, 1), (0, 0), (0, 0)],  # 16
                     [(3, 1), (4, 1), (0, 0)],  # 17
                     [(1, 2), (0, 0), (0, 0)],  # 18
                     [(0, 0), (0, 0), (0, 0)]  # 19
                     ]

    VACP =[
                 0,  # 虚拟1
                 2,  # 2 特设外观检查#供电
                 1.5,  # 3 特设座舱检查
                 1.5,  # 4 航电外观检查
                 1.5,  # 5 航电座舱检查
                 1.5,  # 6 军械外观检查
                 1.5,  # 7 军械座舱检查
                 2,  # 8 航空弹药加载
                 2.5,  # 9 添加燃油
                 2,  # 10 添加液压油
                 2.5,  # 11 充氮
                 1.5,  # 12 机械座舱检查
                 1.5,  # 13 机械外观检查
                 2,  # 14 发动机检查
                 2.5,  # 15 充氧
                 2.5,  # 16 挂弹
                 2.5,  # 17 挂弹
                 1.5, # 18 惯导
                 0 # 19
                 ]

    OrderTime = [
                 0,  # 虚拟1
                 3,  # 2 特设外观检查#供电
                 6,  # 3 特设座舱检查
                 3,  # 4 航电外观检查
                 6,  # 5 航电座舱检查
                 5,  # 6 军械外观检查
                 4,  # 7 军械座舱检查
                 5,  # 8 航空弹药加载
                 13,  # 9 添加燃油
                 4,  # 10 添加液压油
                 4,  # 11 充氮
                 3,  # 12 机械座舱检查
                 12,  # 13 机械外观检查
                 8,  # 14 发动机检查
                 3,  # 15 充氧
                 8,  # 16 挂弹
                 8,  # 17 挂弹
                 7, # 18 惯导
                 0 # 19
                 ]

    lowTime = 120  # 不能超过90 min
    HS = 3
    # Lpk = [0 for _ in range(Human_resource_type)]
    # for p in range(planeNum):
    #     for i in range(0,planeOrderNum):
    #         needRtype = OrderInputMes[i][0][0]
    #         needNums = OrderInputMes[i][0][1]
    #         dur = OrderTime[i]
    #         Lpk[needRtype] += HS*needNums*dur/lowTime
    #
    # Lpk = [int(i) for i in Lpk]
    # total_Human_resource = Lpk
    sigma = 0.3
    shedule_num=0
    act_info={}

    human_walk_speed = 80 #人员行走速度8 m/(in)


    humanNum = 0
    targetWeight =[1,0.3,0.1]
    boundUpper =[0,0]
    boundLowwer=[]
    AON=[]








