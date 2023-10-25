import copy

class Space:
    def __init__(self,id):
        self.pos_id = id
        self.working = False
        self.NowTaskId = 0
        self.OrderOver = []  # 已完成工序
        #已完成工序
        self.TaskWait = [] #待完成工序

    def update(self,Activity):
        self.OrderOver.append(Activity)
      #  self.OrderOver.sort(key=lambda x: x.es)



