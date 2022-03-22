import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import math
import mplcursors
import per_cycle_analysis.single_sim_analysis as ipc
import random as rng
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D



rng.seed(555)

t = np.arange(0,32,0.1) 


def cos_function(time_step,amplitude):
    return (amplitude)*np.cos(time_step*(1/4))+0.5





plt.style.use('fivethirtyeight')



index = iter([1,(1/2),(1/4),(1/8),(1/16),(1/32),0.0])


def get_data(amplitude):
    df=None
    step_size=1
    sensitivity=0.0001
    x = np.arange(4000,4500,0.5) 
    y = [cos_function(time_step=time_step,amplitude=amplitude) for time_step in x]
    yx_dx = ipc.aprox_derv_central(data=y,step_size_forward=1,step_size_backward=step_size)
    smooth_yx_dx = ipc.savitzky_golay(y_list=yx_dx, window_size=3, order=1, deriv=0, rate=1)
    yx_dx2 = ipc.aprox_derv_central(data=yx_dx,step_size_forward=step_size,step_size_backward=step_size)
    smooth_yx_dx2 = ipc.savitzky_golay(y_list=yx_dx2, window_size=3, order=1, deriv=0, rate=1)
    data_tuples = list(zip(x,y,smooth_yx_dx,smooth_yx_dx2))
    df = pd.DataFrame(data_tuples, columns=['x','y','yx_dx','yx_dx2'])
    classifacation = ipc.inflection_point_classifaction(df=df,sensitivity=sensitivity)
    df['point_classifacation'] = classifacation
    dot_class = ipc.plotable_points(df=df)
    df['dot_class'] = dot_class
    df['min_max_flip'] = ipc.min_max_flip_by_gen(df=df)
    #df['time_as_max_min'] = df.groupby((df['min_max_flip'] != df['min_max_flip'].shift(1)).cumsum()).cumcount()+1
    df['time_as_max_min'] = ipc.time_as_min_max(df=df)
    df['min_max_filter'] = ipc.min_max_filter(df=df)
    return df['x'], df['y'], df['dot_class'], df['point_classifacation']


# def animate(i):
#     sns.lineplot(ax=axes,data=df, x="x", y="y", color = 'black')
#     sns.scatterplot(ax=axes,data=df, x="x", y="dot_class", hue="point_classifacation",s=100)
#     axes.set(title='Owl Exp 1: \n Microhabitat Preference Per Generation',
#            ylabel = "Microhabitat Preference",
#            xlabel = "Generation",
#            ylim = (0,1),
#            xlim=(4100, 4150),)





#dots = ax.scatter([], [], c=[])

color_map = {'max':'blue',
             'min':'red'}

fig = plt.figure(figsize=(12,8))
   


def animate(i):
    amp = next(index)
    print(amp)
    x,y,dc,pc = get_data(amplitude=amp)
    #print(dc.unique())
    plt.cla()
    g=sns.lineplot(x=x, y=y, color = 'black', size = 1)
    g1=sns.scatterplot(x=x, y=dc, hue=pc,s=100)
    #plt.plot(x, y, c='black', linewidth=1)
    #plt.scatter(x, list(dc))#, c=pc.map(color_map))
    legend_elements = [Line2D([0], [0], marker='o', color='w', label='min',
                          markerfacecolor='b', markersize=15),
                   Line2D([0], [0], marker='o', color='w', label='max',
                          markerfacecolor='r', markersize=15),]
    plt.xlim(4000, 4500)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Generation')
    plt.ylabel('Microhabitat Preference')
    plt.title('Exp: {}'.format(str(i+1)))
    plt.legend(handles=legend_elements, loc='upper right')

        

    #ax.legend(loc='upper left')
    #plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate, interval=2000)
ani.save('nash.gif', writer='imagemagick')
plt.tight_layout()
plt.show()
# anim = FuncAnimation(fig, animate, init_func=init,
#                                frames=200, interval=20, blit=True)




