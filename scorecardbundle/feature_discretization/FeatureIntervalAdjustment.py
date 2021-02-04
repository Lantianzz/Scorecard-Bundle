# -*- coding: utf-8 -*-
"""
Visualizing feature event rate distribution to facilitate explainability evaluation.

@authors: Lantian ZHANG
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
# Basic functions
# ============================================================

def feature_stat(x,y,delimiter='~'):
    """Compute the input feature's sample distribution, including
    the sample sizes, event sizes and event proportions 
    of each feature value.
    Parameters
    ----------
    x: numpy.array, shape (number of examples,)
        The discretizated feature array. Each value represent a right-closed interval
        of the input feature. e.g. '1~8'
    y: numpy.array, shape (number of examples,)
        The binary dependent variable with 1 represents the target event (positive class).

    delimiter: python string. Default is '~'
        The symbol that separates the boundaries of a interval in array x.

    Returns
    -------
    res: pandas.DataFrame, shape (number of intervals in the feature, 4)
        The feature distribution table.
    """
    idx = [float(e.split(delimiter)[-1]) if e.split(delimiter)[-1] not in ('inf','-inf') else (float('inf') if e.split(delimiter)[-1]=='inf' else -float('inf')) for e in x]
    tem = pd.DataFrame({
        'feature':x
        ,'y':y
        ,'idx':idx
    })
    keys = ['idx','feature']
    res = tem.groupby(keys).size().reset_index().rename(columns={0:'sample_size'})
    res['event_num'] = tem.groupby(keys)['y'].sum().values
    res['event_pct'] = (res['event_num']*100/res['sample_size']).round(2).astype(str)+'%'
    return res.set_index('idx').sort_index()

def feature_stat_str(x, y, delimiter='~', n_lines=40, width=20):
    """Compute the input feature's sample distribution in string format for printing.
    The distribution table returned (in string format) concains the sample sizes, 
    event sizes and event proportions of each feature value.

    Parameters
    ----------
    x: numpy.array, shape (number of examples,)
        The discretizated feature array. Each value represent a right-closed interval
        of the input feature. e.g. '1~8'
    y: numpy.array, shape (number of examples,)
        The binary dependent variable with 1 represents the target event (positive class).

    delimiter: python string. Default is '~'
        The symbol that separates the boundaries of a interval in array x.

    n_lines: integer. Default is 40.
        The number of '- ' used. This Controls the length of horizontal lines in the table. 
    
    width: integer. Default is 20.
        This controls the width of each column.

    Returns
    -------
    table_string: python string
        The feature distribution table in string format
    """
    res = feature_stat(x,y,delimiter) # Compute the feature distrition table
    list_str = [] # String table will be constructed line by line
    # Table header
    for i in range(res.shape[1]):
        list_str.extend([str(res.columns[i]),' '*(width-len(res.columns[i].encode('gbk')))])
    list_str.append('\n(right-closed)')
    list_str.extend(['\n','- '*n_lines,'\n'])
    # Table body
    for i in range(res.shape[0]):
        for j in range(res.shape[1]):
            list_str.extend([str(res.iloc[i,j]),' '*(width-len(str(res.iloc[i,j])))])
        list_str.extend(['\n','- '*n_lines,'\n'])
    # Put everything together
    table_string = ''.join(list_str)
    return table_string

# ============================================================
# Visualization
# ============================================================

def plot_event_dist(x, y, delimiter='~',
                title='', x_label='', y_label='', 
                x_rotation=0, xticks=None, 
                figure_height=4, figure_width=6, 
                table_vpos=None,table_hpos=0.01,
                save=False, path='',file_name='feature'):
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

    # Check the inputs
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

    # Data for plots
    data = pd.DataFrame({
        'x':x,
        'y':y
        })
    data['x_b'] = [float(e.split(delimiter)[-1]) if e.split(delimiter)[-1] not in ('inf','-inf') else (float('inf') if e.split(delimiter)[-1]=='inf' else -float('inf')) for e in data.x]
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

    # Plot
    plt.figure(figsize=(figure_width,figure_height))
    plt.suptitle(title, fontdict=font_title)
    # Plot sample distribution as bar plot
    f1_ax1 = plt.subplot(1, 1, 1)
    f1_ax1.bar(plot_x, freq.values,width=0.2)
    f1_ax2 = f1_ax1.twinx() # the x axis shared by two y axies
    f1_ax2.plot(plot_x, plot_y, 'r', alpha=0.4)
    # Plot event rate distribution as line plot
    f1_ax2.legend(('Event Rate',),loc=1)
    f1_ax1.set_xticks(plot_x) # set xtick value
    f1_ax1.set_xticklabels(plot_x_labels) # set xtick label
    f1_ax1.tick_params(axis='x', rotation=x_rotation) # set xtick label's rotation
    f1_ax1.set_xlabel(x_label, fontdict=font_text)
    f1_ax2.set_ylabel('Event Rate: '+y_label, fontdict=font_text)
    f1_ax1.legend(('Sample size',),loc=2)
    f1_ax1.set_ylabel('Sample size', fontdict=font_text)
    

    # feature distribution table
    tem_stat = feature_stat(x,y) 
    tem_stat_str = feature_stat_str(x,y)
    # The vertical position of table below the plot.
    # Either use the parameter 'table_vpos',
    # or assign a value according to the number of rows in the table.
    if table_vpos is not None:
        pass # Use the value passed to parameter 'table_vpos'
    elif tem_stat.shape[0]<=2:
        table_vpos = -0.45 
    elif tem_stat.shape[0]==3:
        table_vpos = -0.5
    elif tem_stat.shape[0]==4:
        table_vpos = -0.55
    else:
        table_vpos = -0.55*np.log(tem_stat.shape[0]-1)
    # Output feature distribution table along with the plot
    plt.figtext(table_hpos,table_vpos,tem_stat_str)

    # Save plot as png file in a local position
    if save:
        plt.savefig(f'{path}featuredist_{file_name}.png',dpi=500,bbox_inches='tight')

    # Print the feature distribution table
    print(tem_stat_str)
