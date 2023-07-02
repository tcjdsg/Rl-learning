import scipy.stats as stats
from JZJenv.FixedMess import FixedMes

def getTime(j, i):
    # 定义每种任务的时间分布
    sigma = 0.3
    if i == 0:
        return 0
    elif i == 1:
        return 0
    elif i == 2:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[2], scale=sigma).rvs()
    elif i == 3:
        return stats.truncnorm((-0.8) / sigma, 0.8 / sigma, loc=FixedMes.OrderTime[3], scale=sigma).rvs()
    elif i == 4:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[4], scale=sigma).rvs()
    elif i == 5:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[5], scale=sigma).rvs()
    elif i == 5:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[5], scale=sigma).rvs()
    elif i == 6:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[6], scale=sigma).rvs()
    elif i == 7:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[7], scale=sigma).rvs()

    elif i == 8:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[8], scale=sigma).rvs()
    elif i == 9:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[9], scale=sigma).rvs()

    elif i == 10:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[10], scale=sigma).rvs()
    elif i == 11:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[11], scale=sigma).rvs()

    elif i == 12:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[12], scale=sigma).rvs()
    elif i == 13:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[13], scale=sigma).rvs()
    elif i == 14:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[14], scale=sigma).rvs()

    elif i == 15:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[15], scale=sigma).rvs()
    elif i == 16:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[16], scale=sigma).rvs()
    elif i == 17:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[17], scale=sigma).rvs()
    elif i == 18:
        return stats.truncnorm((-0.5) / sigma, 0.5 / sigma, loc=FixedMes.OrderTime[18], scale=sigma).rvs()
    elif i == 19:
        return 0
