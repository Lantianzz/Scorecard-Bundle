# -*- coding: utf-8 -*-
"""
Created on Thu Nov 1 2018
Updated on Sat Mar 31 2020

@authors: Lantian ZHANG <peter.lantian.zhang@outlook.com>

Visualizing feature event rate distribution to facilitate explainability evaluation.
"""
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

# ============================================================
# Basic plot settings
# ============================================================
plt.style.use('seaborn-colorblind')
plt.rcParams['font.sans-serif'] = ['SimHei']  # Enable display of Chinese characters
plt.rcParams['axes.unicode_minus'] = False  # Enable display of negative sign '-'

plt.rcParams['savefig.dpi'] = 300 
plt.rcParams['figure.dpi'] = 150

# Font settings
font_text = {'family':'SimHei', 
        'weight':'normal',
         'size':11,
        } # Font setting for normal texts

font_title = {'family':'SimHei',
        'weight':'bold',
         'size':14,
        } # Font setting for title

# Thousands seperator
from matplotlib.ticker import FuncFormatter 
def format_thousands(x,pos):
    return '{:,.0f}'.format(x,pos)
formatter_thousands = FuncFormatter(format_thousands)

# ============================================================
# Visualization
# ============================================================

def plot_event_dist(x, y, delimiter='~',
                title='', x_label='', y_label='', 
                x_rotation=0, xticks=None, 
                figure_height=4, figure_width=6, 
                save=False, path=''):
    """Visualizing feature event rate distribution 
    to facilitate explainability evaluation.
    
    Parameters
    ----------
    x:numpy.ndarray or pandas.DataFrame, shape (number of examples,)
        The feature to be visualized.
    
    y:numpy.ndarray or pandas.DataFrame, shape (number of examples,)
        The Dependent variable.

    delimiter: string, optional(default='~')
        The interval is representated by string (i.e. '1~2'), 
        which takes the form lower+delimiter+upper. This parameter 
        control the symbol that connects the lower and upper boundaries.   

    title: Python string. Optional.
        The title of the plot. Default is ''.
    
    x_label: Python string. Optional.
        The label of the feature. Default is ''.
    
    y_label: Python string. Optional.
        The label of the dependent variable. Default is ''.
    
    x_rotation: int. Optional.
        The degree of rotation of x-axis ticks. Default is 0.
    
    xticks: Python list of strings. Optional.
        The tick labels on x-axis. Default is the unique values
        of x (in the format of Python string).
    
    figure_height: int. Optional.
        The hight of the figure. Default is 4.
    
    figure_width: int. Optional.
        The width of the figure. Default is 6.
    
    save: boolean. Optional.
        Whether or not the figure is saved to a local positon.
        Default is False.
    
    path: Python string. Optional.
        The local position path where the figure will be saved.
        This should be set when parameter save is True. Default is ''.
    
        
    Returns
    -------
    f1_ax1: matplotlib.axes._subplots.AxesSubplot
        The figure object is returned.
    """    
    if isinstance(x, pd.Series):
        x = x.values
    elif isinstance(x, np.ndarray):
        pass
    else:
        raise TypeError('x should be either numpy.array or pandas.Series')

    if isinstance(y, pd.Series):
        y = y.values
    elif isinstance(y, np.ndarray):
        pass
    else:
        raise TypeError('y should be either numpy.array or pandas.Series')

    data = pd.DataFrame({
        'x':x,
        'y':y
        })
    ' -inf~34.49'.split('~')[-1]
    data['x_b'] = [float(e.split('~')[-1]) if e.split('~')[-1] not in ('inf','-inf') else (float('inf') if e.split('~')[-1]=='inf' else -float('inf')) for e in data.x]
    map_xticks = data[['x','x_b']].drop_duplicates().sort_values('x_b')
    event = data.groupby('x_b')['y'].sum()
    freq = data.groupby('x_b')['y'].size()
    plot_data = event/freq 
    plot_x = np.arange(len(plot_data))
    plot_y = plot_data.values

    if xticks is None:
        plot_x_labels = map_xticks['x']
    else:
        plot_x_labels = xticks

    plt.figure(figsize=(figure_width,figure_height))
    plt.suptitle(title, fontdict=font_title)
    
    f1_ax1 = plt.subplot(1, 1, 1)
    f1_ax1.bar(plot_x, freq.values,width=0.2)
    f1_ax2 = f1_ax1.twinx() # the x axis shared by two y axies
    f1_ax2.plot(plot_x, plot_y, 'r', alpha=0.4)
    
    f1_ax2.legend(('Event Rate',),loc=1)
    f1_ax1.set_xticks(plot_x) # set xtick value
    f1_ax1.set_xticklabels(plot_x_labels) # set xtick label
    f1_ax1.tick_params(axis='x', rotation=x_rotation) # set xtick label's rotation
    f1_ax1.set_xlabel(x_label, fontdict=font_text)
    f1_ax2.set_ylabel('Event Rate: '+y_label, fontdict=font_text)
    f1_ax1.legend(('Sample size',),loc=2)
    f1_ax1.set_ylabel('Sample size', fontdict=font_text)
    
    if save:
        plt.savefig(path+x_label+y_label+'.png',dpi=500,bbox_inches='tight')

    res = event.reset_index().rename(columns={'y':'event_freq'})
    res['sample_size'] = freq.values
    res['event_rate'] = plot_data.values
    print(res)
    return f1_ax1
