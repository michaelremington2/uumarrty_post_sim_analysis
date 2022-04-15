#!/usr/bin/python
import pandas as pd
import numpy as np
import classification_accuracy as ca 
import matplotlib.pyplot as plt
import seaborn as sns 
import math


snake_exp1_optima = [(1504,1.00,'max'),
                     (1507,0.033,'min'),
                     (1510,0.990,'max'),
                     (1513,0.011,'min'),
                     (1515,0.977,'max'),
                     (1517,0.032,'min'),
                     (1519,0.989,'max'),
                     (1521,0.000,'min'),
                     (1524,0.989,'max'),
                     (1526,0.035,'min'),
                     (1528,0.956,'max'),
                     (1530,0.024,'min'),
                     (1532,0.988,'max'),
                     (1535,0.024,'min'),
                     (1537,1.000,'max'),
                     (1539,0.000,'min'),
                     (1541,0.989,'max'),
                     (1543,0.012,'min'),
                     (1545,0.989,'max'),
                     (1547,0.000,'min'),
                     (1550,1.000,'max'),
                     (1553,0.011,'min'),
                     (1555,0.989,'max'),
                     (1557,0.034,'min'),
                     (1559,0.977,'max'),
                     (1561,0.021,'min'),
                     (1563,0.987,'max'),
                     (1565,0.000,'min'),
                     (1572,0.977,'max'),
                     (1574,0.000,'min'),
                     (1572,0.977,'max'),
                     (1574,0.000,'min'),
                     (1577,0.988,'max'),
                     (1579,0.010,'min'),
                     (1582,0.994,'max'),
                     (1585,0.122,'min'),
                     (1586,0.964,'max'),
                     (1588,0.021,'min'),
                     (1590,0.979,'max'),
                     (1592,0.011,'min'),
                     (1594,1.000,'max'),
                     (1597,0.022,'min'),
                     (1599,0.989,'max'),
                     (1601,0.234,'min'),
                     (1603,0.967,'max'),
                     (1605,0.025,'min'),
                     (1607,0.987,'max'),
                     (1609,0.032,'min'),
                     (1611,0.990,'max'),
                     (1613,0.004,'min'),
                     (1617,0.958,'max'),
                     (1619,0.011,'min'),
                     (1622,1.000,'max'),
                     (1625,0.011,'min'),
                     (1627,0.979,'max'),
                     (1629,0.023,'min'),
                     (1631,0.935,'max'),
                     (1634,0.000,'min'),
                     (1637,0.964,'max'),
                     (1639,0.011,'min'),
                     (1641,1.000,'max'),
                     (1649,0.000,'min'),
                     (1653,1.000,'max'),
                     (1657,0.000,'min'),
                     (1660,0.996,'max')]

krat_exp2_optima=[(9535,0.800,'max'),
                 (9562,0.122,'min'),
                 (9580,0.814,'max'),
                 (9601,0.132,'min'),
                 (9623,0.878,'max'),
                 (9648,0.127,'min'),
                 (9677,0.848,'max'),
                 (9701,0.158,'min'),
                 (9727,0.866,'max'),
                 (9754,0.197,'min'),
                 (9773,0.691,'max'),
                 (9796,0.142,'min'),
                 (9815,0.878,'max'),
                 (9845,0.132,'min'),
                 (9870,0.813,'max'),
                 (9899,0.112,'min'),
                 (9929,0.837,'max'),
                 (9949,0.144,'min'),
                 (9972,0.876,'max')]

####################
## Derivative aprox
####################

def aprox_derv_forward(data,step_size):
    derv = []
    for i in range(len(data)-step_size):
        val = round(((data[i+step_size]-data[i])/step_size),2)
        derv.append(val)
    return derv

def aprox_derv_backward(data,step_size):
    derv = []
    for i in range(len(data)):
        val = (data[i]-data[i-step_size])/step_size
        derv.append(val)
    return derv

def aprox_derv_central(data,step_size_forward,step_size_backward):
    derv = []
    if step_size_backward<0:
        raise ValueError()
    for i in range(len(data)):
        try:
            if i-step_size_backward<0:
                val = float("NAN")
            else:
                val = (data[i+step_size_forward]-data[i-step_size_backward])/(step_size_forward+step_size_backward)
        except IndexError:
            val = float("NAN")
        derv.append(val)
    return derv

def aprox_derv_forward_v2(data,step_size):
    return np.diff(data,1)/step_size

def savitzky_golay(y_list, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial
    
    try:
        y = np.asarray(y_list)
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order+1)
    half_window = (window_size -1) // 2
    # precompute coefficients
    b = np.mat([[k**i for i in order_range] for k in range(-half_window, half_window+1)])
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs( y[1:half_window+1][::-1] - y[0] )
    lastvals = y[-1] + np.abs(y[-half_window-1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve( m[::-1], y, mode='valid')

def inflection_point_classifaction(df,sensitivity=0.1):
    classifacation = []
    max_id=df['x'].idxmax()
    for index, row in df.iterrows():
        try:
            no_key_error=bool((index+1)<=max_id)
            index_not_zero=bool(index!=0)
            if no_key_error and index_not_zero:
#                 sig_max_check=(abs(df.loc[int(index+step_size), 'yx_dx'])+abs(df.loc[int(index-step_size), 'yx_dx']))/(step_size*2+1)>sensitivity
#                 sig_min_check=(abs(df.loc[int(index+step_size), 'yx_dx'])+abs(df.loc[int(index-step_size), 'yx_dx']))/(step_size*2+1)>sensitivity
                sig_max_check=abs(df.loc[int(index+1), 'yx_dx2'])>sensitivity
                sig_min_check=abs(df.loc[int(index+1), 'yx_dx2'])>sensitivity
                if round(row['yx_dx'],2)==0.0 and round(row['yx_dx2'],2)>0.0 and sig_min_check:
                    val='min'
                elif round(row['yx_dx'],2)==0.0 and round(row['yx_dx2'],2)<0.0 and sig_max_check:
                    val='max'
                elif df.loc[int(index+1), 'yx_dx']>0 and df.loc[int(index), 'yx_dx']<0 and sig_min_check:
                    val='min'
                elif df.loc[int(index+1), 'yx_dx']<0 and df.loc[int(index), 'yx_dx']>0 and sig_max_check:
                    val='max'
                else:
                    val=float('NAN')
            else:
                val=float('NAN')
        except KeyError:
            val=float('NAN')
        classifacation.append(val)
    return classifacation

def plotable_points(df):
    dot_class = []
    for index, row in df.iterrows():
        if row['point_classifacation'] != float('NAN'):
            val=row['y']
        else:
            val=float('NAN')
        dot_class.append(val)
    return dot_class
### EXP1
file_path_per_cycle = '/home/mremington/Documents/uumarrty_exps/owl_exp/exp1/Data/per_cycle.csv'
file_path_parameter = '/home/mremington/Documents/uumarrty_exps/owl_exp/exp1/Data/parameters.csv'
per_cycle = pd.read_csv(file_path_per_cycle,header = 0, index_col=None)
parameters = pd.read_csv(file_path_parameter,header = 0, index_col=None)
sim = per_cycle[per_cycle['sim_number']==0]
sim = sim[sim['experiment']=='experiment1']
sim_krat = sim[sim['org']=='krat']
sim_snake = sim[sim['org']=='snake']
krat_reproduction_freq = parameters['krat_reproduction_freq_per_x_cycles'].max()
snake_reproduction_freq = parameters['snake_reproduction_freq_per_x_cycles'].max()
#### EXP 2 ######
file_path_per_cycle_exp2 = '/home/mremington/Documents/uumarrty_exps/mixed_owl_exp/exp2/Data/per_cycle.csv'
file_path_parameter_exp2 = '/home/mremington/Documents/uumarrty_exps/mixed_owl_exp/exp2/Data/parameters.csv'
per_cycle_exp2 = pd.read_csv(file_path_per_cycle_exp2,header = 0, index_col=None)
parameters_exp2 = pd.read_csv(file_path_parameter_exp2,header = 0, index_col=None)
sim_exp2 = per_cycle_exp2[per_cycle_exp2['sim_number']==0]
sim_exp2 = sim_exp2[sim_exp2['experiment']=='experiment2']
sim_krat_exp2 = sim_exp2[sim_exp2['org']=='krat']
sim_snake_exp2 = sim_exp2[sim_exp2['org']=='snake']
krat_reproduction_freq_exp2 = parameters_exp2['krat_reproduction_freq_per_x_cycles'].max()
snake_reproduction_freq_exp2 = parameters_exp2['snake_reproduction_freq_per_x_cycles'].max()
krat_by_gen_exp2 = sim_krat_exp2[sim_krat_exp2.apply(lambda sim_krat_exp2: (sim_krat_exp2['cycle'] % krat_reproduction_freq_exp2)==0, axis=1)]
snake_by_gen_exp2 = sim_snake_exp2[sim_snake_exp2.apply(lambda sim_snake_exp2: (sim_snake_exp2['cycle'] % snake_reproduction_freq_exp2)==0, axis=1)]


### Snakes ####
#exp1
#rolling_mean_window_size = 3

#krat_by_gen['bush_pw_ma']=krat_by_gen['bush_pw_mean'].rolling(window=rolling_mean_window_size,center=False).mean()
snake_by_gen = sim_snake[sim_snake.apply(lambda sim_snake: (sim_snake['cycle'] % snake_reproduction_freq)==0, axis=1)]
#snake_by_gen['bush_pw_ma']=snake_by_gen['bush_pw_mean'].rolling(window=rolling_mean_window_size,center=False).mean()
step_size=1
sensitivity=0.03
#np.over_cursers
snake_gen_temp = snake_by_gen[snake_by_gen['generation']>=1500]
#snake_gen_temp = snake_by_gen_exp3[snake_by_gen_exp3['generation']>=1600]
x = list(snake_gen_temp['generation'])
y = list(snake_gen_temp["bush_pw_mean"])
yx_dx = aprox_derv_central(y,step_size_forward=1,step_size_backward=step_size)
smooth_yx_dx = savitzky_golay(y_list=yx_dx, window_size=3, order=1, deriv=0, rate=1)
yx_dx2 = aprox_derv_central(yx_dx,step_size_forward=step_size,step_size_backward=step_size)
smooth_yx_dx2 = savitzky_golay(y_list=yx_dx2, window_size=3, order=1, deriv=0, rate=1)
#data_tuples = list(zip(x,y,yx_dx,smooth_yx_dx,yx_dx2))
data_tuples = list(zip(x,y,smooth_yx_dx,smooth_yx_dx2))
df_snake = pd.DataFrame(data_tuples, columns=['x','y','yx_dx','yx_dx2'])

classifacation = inflection_point_classifaction(df=df_snake,sensitivity=sensitivity)
df_snake['point_classifacation'] = classifacation
dot_class = plotable_points(df=df_snake)
df_snake['dot_class'] = dot_class

###### Krat exp2######
krat_by_gen = sim_krat[sim_krat.apply(lambda sim_krat: (sim_krat['cycle'] % krat_reproduction_freq)==0, axis=1)]
step_size=1
sensitivity=.01
window_size=5
order=1
deriv=0
rate=1


#krat_by_gen_temp = krat_by_gen[krat_by_gen['generation']>=9500]
krat_by_gen_temp = krat_by_gen_exp2[krat_by_gen_exp2['generation']>=9500]
x = list(krat_by_gen_temp['generation'])
y = list(krat_by_gen_temp["bush_pw_mean"])
yx_dx = aprox_derv_central(y,step_size_forward=step_size,step_size_backward=step_size)
smooth_yx_dx = savitzky_golay(y_list=yx_dx, window_size=window_size, order=order, deriv=deriv, rate=rate)
yx_dx2 = aprox_derv_central(yx_dx,step_size_forward=step_size,step_size_backward=step_size)
smooth_yx_dx2 = savitzky_golay(y_list=yx_dx2, window_size=window_size, order=order, deriv=deriv, rate=rate)
#data_tuples = list(zip(x,y,yx_dx,smooth_yx_dx,yx_dx2))
data_tuples = list(zip(x,y,smooth_yx_dx,smooth_yx_dx2))
df_krat = pd.DataFrame(data_tuples, columns=['x','y','yx_dx','yx_dx2'])

classifacation = inflection_point_classifaction(df=df_krat,sensitivity=sensitivity)
df_krat['point_classifacation'] = classifacation
dot_class = plotable_points(df=df_krat)
df_krat['dot_class'] = dot_class

####################
def main_snake(x_range=1,y_range=0.01,padded=False):
    temp = ca.CalcPerCycleAccuracy(actual_optima_points=snake_exp1_optima,
                              x_range = x_range,
                              y_range = y_range,
                              test_set=df_snake,
                              padded=padded)
    temp.main()
    accuracy=temp.calc_accuracy(true_positive_list=temp.true_positives,true_negative_list=temp.true_negatives,test_set=df_snake)
    precision=temp.calc_precision(true_positive_list=temp.true_positives,false_positive_list=temp.false_positives)
    recall=temp.calc_recall(true_positive_list=temp.true_positives,false_negaitve_list=temp.false_negatives)
    print('___________________________')
    temp.test_confusion_matrix()
    print('______Confusion Matrix_____')
    temp.print_confusion_matrix_total()
    print('___________________________')
    print('Accuracy total {}'.format(accuracy))
    print('Precision total {}'.format(precision))
    print('Recall total {}'.format(recall))

def main_krat(x_range=1,y_range=0.01,padded=False):
    temp = ca.CalcPerCycleAccuracy(actual_optima_points=krat_exp2_optima,
                              x_range = x_range,
                              y_range = y_range,
                              test_set=df_krat,
                              padded=padded)
    temp.main()
    accuracy=temp.calc_accuracy(true_positive_list=temp.true_positives,true_negative_list=temp.true_negatives,test_set=df_krat)
    precision=temp.calc_precision(true_positive_list=temp.true_positives,false_positive_list=temp.false_positives)
    recall=temp.calc_recall(true_positive_list=temp.true_positives,false_negaitve_list=temp.false_negatives)
    print('___________________________')
    temp.test_confusion_matrix()
    print('______Confusion Matrix_____')
    temp.print_confusion_matrix_total()
    print('___________________________')
    print('Accuracy total {}'.format(accuracy))
    print('Precision total {}'.format(precision))
    print('Recall total {}'.format(recall))

if __name__ == "__main__":
    #main_snake(x_range=1,y_range=0.01,padded=False)
    main_krat(x_range=1,y_range=0.01,padded=False)

