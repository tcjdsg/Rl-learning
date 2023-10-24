
class Order(object):
    '''
    活动类：包含  1.活动总ID  、舰载机活动Id，舰载机Id
    2.活动持续时间    3.活动资源需求量   4.活动紧前活动
     5.活动最早开始时间  6.活动最晚开始时间  7.活动是否被访问
    '''

    def __init__(self, id,taskid, duration, RequestHumanType, needHuman, RequestStationType,  successor, jzjId):
        self.id = id
        self.taskid = taskid
        self.belong_plane_id = jzjId
        self.vacp = 0

        #入场时间
        self.ST = 0
        self.duration = duration
        self.resourceRequestH = RequestHumanType
        self.needH = needHuman
        self.RequestStationType = RequestStationType

        self.predecessor = None
        self.successor = successor
        self.es = 0
        self.ef = 0
        self.ls = 0
        self.lf = 0
        self.tf = 0
        self.mts =0
        self.grpw = 0
        self.grd=0
        self.exist = True
        self.dur = 0

        self.acs=0
        self.wcs=0
        self.irsm=0

        self.scheduled = False
        self.complete = False
        self.HumanReady = True
        self.working = False
        self.priority = 0

        self.TimeStart = 0
        self.TimeEnd = self.TimeStart + self.duration
        self.HumanNums = [] #执行任务的人员编号
        self.SheiBei = []
        self.SNums = [] #执行任务编号

    def __lt__(self, other):
        return self.es < other.es
    # def __eq__(self, other):
    #     return self.es == other.es
    def __gt__(self, other):
        return self.es > other.es





