from collections import defaultdict



class FixedMes(object):
    """
    distance:
    orderInputMes:
    """
    pri = ['LFT','LST','FIFO','RAND','SPT','MTS','GRPW','GRD','IRSM','WCS','ACS']
    #全局时钟
    t = 0


    distance = [[]]

    numJzjPos = 18
    numHumanAll = [18,60]

    planeOrderNum = 19
    planeNum = 8
    jzjNumbers=[1,2,3,4,5,6,7,8]  #舰载机编号
    jzjStartTime = [0,1,2,3,4,5,6,7] #舰载机入场时间

    #座舱限制。相当于是每个站位都有一个座舱，每个舰载机只能用自己座舱。
    space_resource_type = planeNum
    total_space_resource = [1 for i in range(planeNum)]
    Human_resource_type = 4
    # 特设、航电、军械、机械
    # total_Huamn_resource = [4,5,6,8]  # 每种人员数量

    # total_Huamn_resource = [30]
    constraintOrder = defaultdict(lambda: []) #记录每类人的可作用工序，和可作用舰载机范围
    # constraintOrder[0] = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]

    constraintOrder[0] = [ 1, 2, 5]
    constraintOrder[1] = [3,4, 16,17]
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

    Activity_num  = (planeOrderNum)*planeNum #活动数量
    #工序顺序
    SUCOrder = defaultdict(lambda: [])
    SUCOrder[0] = [1,3,5,8,9,10,11,12,13]
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
    SUCOrder[14] = [15,16]
    SUCOrder[15] = [17]
    SUCOrder[16] = [17]
    SUCOrder[17] = [18]
    SUCOrder[18] = []
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
    # OrderTime = [0,
    #              0,  # 虚拟1
    #              3,  # 2 特设外观检查
    #              6,  # 3 特设座舱检查
    #              3,  # 4 航电外观检查
    #              6,  # 5 航电座舱检查
    #              5,  # 6 充氧
    #              4,  # 7 加油
    #              5,  # 8 军械外观检查
    #              13,  # 9 军械座舱检查
    #              4,  # 10 机械座舱检查
    #              4,  # 11 充氮
    #              3,  # 12 机械外观检查
    #              12,  # 13 发动机检查
    #              8,  # 14 机翼展开
    #              3,  # 15 挂弹
    #              8,  # 16 挂弹
    #              10,  # 17 收机翼
    #              7, # 18 惯导
    #              0 # 19
    #              ]
    lowTime = 120  # 不能超过90 min
    HS = 3
    Lpk = [0 for _ in range(Human_resource_type)]
    for p in range(planeNum):
        for i in range(0,planeOrderNum):
            needRtype = OrderInputMes[i][0][0]
            needNums = OrderInputMes[i][0][1]
            dur = OrderTime[i]
            Lpk[needRtype] += HS*needNums*dur/lowTime

    Lpk = [int(i) for i in Lpk]
    total_Human_resource = Lpk
    # total_Huamn_resource = [6,9,12,15]
    # OrderInputMes = [[],
    #                  [(0, 0), (0, 0), (0, 0)],  # 虚拟1
    #                  [(0, 1), (0, 0), (0, 0)],  # 2
    #                  [(0, 1), (1, 1), (0, 1)],  # 3
    #                  [(0, 1), (0, 0), (0, 0)],  # 4
    #                  [(0, 1), (1, 1), (0, 0)],  # 5
    #                  [(0, 1), (0, 0), (0, 0)],  # 6
    #                  [(0, 2), (1, 1), (0, 1)],  # 7
    #                  [(0, 2), (0, 0), (0, 0)],  # 8,
    #                  [(0, 1), (0, 1), (0, 1)],  # 9
    #                  [(0, 1), (4, 1), (0, 0)],  # 10
    #                  [(0, 1), (3, 1), (0, 0)],  # 11
    #                  [(0, 2), (1, 1), (0, 1)],  # 12
    #                  [(0, 1), (0, 0), (0, 0)],  # 13
    #                  [(0, 1), (0, 0), (0, 1)],  # 14
    #                  [(0, 1), (2, 1), (0, 0)],  # 15
    #                  [(0, 1), (0, 0), (0, 0)],  # 16
    #                  [(0, 1), (0, 0), (0, 0)],  # 17
    #                  [(0, 2), (0, 0), (0, 0)],  # 18
    #                  [(0, 0), (0, 0), (0, 0)]  # 19
    #                  ]


    #17位 为了让虚拟从1开始
    sigma = 0.3
    shedule_num=0
    act_info={}


    cross = 0.5
    cross1 = 2.5
    MutationRate = 0.25
    MutationRatePmo = 0.05

    transferrate = 0.2
    transfer_iter = 50
    human_walk_speed = 800000000 #人员行走速度8 m/(in)

    populationnumber = 40
    ge = 100


    threadNum = 1
    populationnumberson = populationnumber

    AgenarationIten = ge / 3
    GenarationIten = 0

    #保存每代染色体信息 父代
    AllFit = []
    AllFitSon = []
    AllFitFamily = []
    #vnsIter = -1

    resver_k1 = [ 0 for _ in range(ge)]
    resver_k2 = [ 0 for _ in range(ge)]
    #populationnumber*populationnumber
    slect_F_step_alone = [[] for _ in range(populationnumber)]
    # slect_F_step = [[] for _ in range(populationnumber)]

    Paternal = [[0,0] for _ in range(int(populationnumber/2))]
    #每一代的平均值
    Avufit = {}
    BestCmax = {}
    BestPr = {}
    BestEcmax = {}
    Bestzonghe = {}
    var ={}
    f = {}
    d = {}
    m = {}

    AverPopmove = 0
    AverPopTime = 0
    AverPopVar = 0
    Diversity = 0.0
    keyChainOrder = []
    #死锁辅助检查列表
    # dealLockList=[[0 for _ in range(Activity_num)] for _ in range(Activity_num)]

    bestHumanNumberTarget=[]

    Allactivity = []
    constraintHuman =[]
    constraintStation=[]
    constraintSpace = []

    humanNum = 0
    targetWeight =[1,0.3,0.1]
    boundUpper =[0,0]
    boundLowwer=[]





    AON=[]


    # import scipy.stats as stats
    # mu, sigma = 5, 0.7
    # lower, upper = mu - 2 * sigma, mu + 2 * sigma  # 截断在[μ-2σ, μ+2σ]
    # X = stats.truncnorm((lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma)
    # print(X.rvs())
    # print(X.rvs())
    #
    # x=FixedMes()
    # x.my()
    # print()







