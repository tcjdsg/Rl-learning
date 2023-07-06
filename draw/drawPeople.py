from matplotlib import pyplot as plt

from JZJenv.FixedMess import FixedMes


def Draw_gantt(all_people):
    colors = ['b', 'c', 'g', 'k', 'm', 'r', 'y', 'grey','0.1','0.2','0.3','0.4','0.5','0.6','0.7','b']
    number=0
    for i in range(len(all_people)):
        for j in range(len(all_people[i])):
            number += 1
            for order in all_people[i][j].OrderOver:
                job = order.belong_plane_id
                gongxu = order.taskid
                time1= order.es
                time2= order.ef
                if (time2 - time1) != 0:
                   plt.barh(number, time2 - time1-0.1,
                     left=time1, color=colors[job-1])
                news = str(gongxu-1)
                infmt = '(jzj' +str(job)+'-'+ news + ')'
                if (time2 - time1)!=0:
                   plt.text(x=time1, y=number-0.1 , s=infmt, fontsize=8,
                       color='white')

    # label_name = ['JOB' + str(i) for i in FixedMes.jzjNumbers]
    # patches = [mpatches.Patch(color=colors[1], label=label_name[0])]
    # plt.legend(handles=patches, loc=4)
    # y = range(1, FixedMes.Human1_resource_type+FixedMes.Human2_resource_type+1, 1)
    # name = ["电氧氮","液压\加油","通风","挂弹","机械","航电","军械","特设"]
    #1充电
    #2氧气
    #3氮气
    #4液压
    #5空调通风
    #6加油
    #7挂弹

    # plt.yticks([i + 1 for i in range(people_number)])
    plt.show()