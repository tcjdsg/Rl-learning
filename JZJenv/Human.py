import copy

from conM.FixedMess import FixedMes


class Human:
    def __init__(self,info):
        self.type = info[0]
        self.zunumber = info[1]
        self.number = info[2]
        self.state = False
        self.NowJZJ = 0
        self.NowTaskId = 0
        self.walkState = False
        self.rest = 0.75


        self.fatigue = 0
        self.record_fatigue = []
        self.alreadyworkTime = 0
        self.OrderOver = [] #已完成工序
        self.TaskWait = [] #待完成工序
        self.WalkTask = [] #走

    def update(self, Activity):
        self.OrderOver.append(copy.deepcopy(Activity))
        self.OrderOver.sort(key=lambda x: x.es)

        temp=[]
        self.record_fatigue=[[0,0]]
        self.fatigue =0

        for i in range(len(self.OrderOver)):
            activ = self.OrderOver[i]
            temp.append(activ)
            vacp = activ.vacp
            if i==0:
                self.fatigue += (activ.es - 0) * self.rest
            else:
                self.fatigue += (activ.es - temp[-2].ef) * self.rest
            self.record_fatigue.append([activ.es, self.fatigue])
            self.fatigue += activ.duration * vacp
            self.record_fatigue.append([activ.ef, self.fatigue])

        self.alreadyworkTime += Activity.duration

    def getfatigue(self, time):
        # fatigue = self.fatigue+self.rest * (time - self.OrderOver[-1].ef)
        return self.fatigue

    def getmovetime(self):
        time = 0
        self.OrderOver.sort(key=lambda x: x.es)
        if len(self.OrderOver) >= 2:
         for index in range(len(self.OrderOver)-1):

            from_pos = self.OrderOver[index].belong_plane_id
            to_pos = self.OrderOver[index+1].belong_plane_id
            # movetime1 = 0 if from_pos == 0 else FixedMes.distance[from_pos][now_pos] * FixedMes.human_walk_speed
            movetime = 0 if to_pos == 0 else FixedMes.distance[from_pos][to_pos]/FixedMes.human_walk_speed
            time+=movetime
        return time


class Humanwalk:
    def __init__(self):
        self.walk_State = False
        self.from_jzj = None
        self.to_jzj = None
        self.walk_s_time = None
        self.walk_distance = None
        self.walk_time = None
        self.walk_speed = 50

