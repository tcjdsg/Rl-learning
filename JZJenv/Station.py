import copy

class Station:
    def __init__(self,info):
        self.type = info[0]
        self.zunumber = info[1]
        self.number = info[2]
        self.working = False
        self.NowJZJ = 0
        self.NowTaskId = 0


        self.alreadyworkTime = 0
        self.OrderOver = []  # 已完成工序
        #已完成工序
        self.TaskWait = [] #待完成工序


    def update(self,Activity):
        self.alreadyworkTime += Activity.duration
        self.OrderOver.append(copy.deepcopy(Activity))
        self.OrderOver.sort(key=lambda x: x.es)



