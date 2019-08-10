# -*- coding: utf-8 -*-
"""
Created on Wed Dec 12 13:18:36 2018
Updated on Thu Dec  13 15:35:00 2018
@author: zhanglt

Model evaluation for binary classification problem.
"""

import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
import numpy as np

plt.style.use('seaborn-colorblind')
plt.rcParams['font.sans-serif'] = ['SimHei'] #解决中文显示问题，设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False #解决保存图像负号'-'显示为方块的问题
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 120

font_text = {'family':'SimHei',
        'weight':'normal',
         'size':12,
        } # font for notmal text

font_title = {'family':'SimHei',
        'weight':'bold',
         'size':16,
        } # font for title


# KS
def ks_stat(y_true, y_pred_proba):
    """calculate the KS of a model
    Parameters
    ----------
    y_true: numpy.array, shape (number of examples,)
            The target column (or dependent variable).  
    
    y_pred_proba: numpy.array, shape (number of examples,)
            The score or probability output by the model. The probability
            of y_true being 1 should increase as this value
            increases.              
    """
    get_ks = lambda y_true, y_pred: scipy.stats.ks_2samp(y_pred[y_true==1], y_pred[y_true!=1]).statistic
    return get_ks(y_true, y_pred_proba) 

def ks(y_true, y_pred_proba, output_path=None):
    """Plot K-S curve of a model
    Parameters
    ----------
    y_true: numpy.array, shape (number of examples,)
            The target column (or dependent variable).  
    
    y_pred_proba: numpy.array, shape (number of examples,)
            The score or probability output by the model. The probability
            of y_true being 1 should increase as this value
            increases.    
    
    output_path: the location to save the plot. Default is None.    
    """    
    interval_index = pd.IntervalIndex(pd.qcut(pd.Series(y_pred_proba).sort_values(ascending=False), 10, duplicates='drop').drop_duplicates()) 
    group = pd.Series([interval_index.get_loc(element) for element in y_pred_proba])
    distribution = pd.DataFrame({
            'group':group,
            'y_true':y_true
            })
    grouped = distribution.groupby('group')
    pct_of_target = grouped['y_true'].sum() / np.sum(y_true)
    pct_of_nontarget = (grouped['y_true'].size() - grouped['y_true'].sum()) / (len(y_true) - np.sum(y_true))
    cumpct_of_target = pd.Series([0] + list(pct_of_target.cumsum()))
    cumpct_of_nontarget = pd.Series([0] + list(pct_of_nontarget.cumsum()))
    diff = cumpct_of_target - cumpct_of_nontarget
    plt.plot(cumpct_of_target, label='Y=1')
    plt.plot(cumpct_of_nontarget, label='Y=0')
    plt.plot(diff, label='K-S curve')
    plt.annotate(s='KS = '+str(round(diff.max(),3)) ,xy=(diff.idxmax(),diff.max()))
    plt.xlim((0,10))
    plt.ylim((0,1))
    plt.title('K-S曲线', fontdict=font_title)   
    plt.xlabel('Y=1的概率的分组', fontdict=font_text)
    plt.ylabel('累计Y=1(或0)占全部Y=1(或0)的比例', fontdict=font_text)    
    plt.legend()

    if output_path is not None:
        plt.savefig(output_path+r'KS曲线.png',dpi=500,bbox_inches='tight')    
    plt.show()
        
# ROC曲线
def roc(y_true, y_pred_proba, output_path=None):
    """Plot ROC curve. Credit to Aurélien Géron's book
    "Hands on Machine Learning with Scikit-learn and Tensorflow".
    
    Parameters
    ----------
    y_true: numpy.array, shape (number of examples,)
            The target column (or dependent variable).  
    
    y_pred_proba: numpy.array, shape (number of examples,)
            The score or probability output by the model. The probability
            of y_true being 1 should increase as this value
            increases.    
    
    output_path: the location to save the plot. Default is None.    
    """  
    print('AUC:',roc_auc_score(y_true, y_pred_proba)) #AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate',fontdict=font_text)
    plt.ylabel('True Positive Rate',fontdict=font_text)
    plt.annotate(s='AUC = '+str(round(roc_auc_score(y_true, y_pred_proba),3)), 
                 xy = (0.03, 0.95))
    plt.title('ROC曲线', fontdict=font_title)

    if output_path is not None:
        plt.savefig(output_path+r'ROC曲线.png',dpi=500,bbox_inches='tight')
    plt.show()

# 精确度vs敏感度曲线
def precision_recall(y_true, y_pred_proba, output_path=None):
    """precision and recall curves. Credit to Aurélien Géron's book
    "Hands on Machine Learning with Scikit-learn and Tensorflow".
    
    Parameters
    ----------
    y_true: numpy.array, shape (number of examples,)
            The target column (or dependent variable).  
    
    y_pred_proba: numpy.array, shape (number of examples,)
            The score or probability output by the model. The probability
            of y_true being 1 should increase as this value
            increases.    
    
    output_path: the location to save the plot. Default is None.    
    """  
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_pred_proba)
    plt.plot(thresholds, precisions[:-1], 'b--', label='精确度（Precision）')
    plt.plot(thresholds, recalls[:-1], 'g-', label='敏感度（Recall）')
    plt.xlabel('阈值（Threshold）', fontdict=font_text)
    plt.legend(loc='center left')
    plt.title('精确度vs敏感度曲线', fontdict=font_title)

    if output_path is not None:
        plt.savefig(output_path+r'精确度vs敏感度曲线.png',dpi=500,bbox_inches='tight')
    plt.show()

def lift_curve(y_true, y_pred_proba, output_path=None):
    """Lift curve and cumulated lift curve.
    
    Parameters
    ----------
    y_true: numpy.array, shape (number of examples,)
            The target column (or dependent variable).  
    
    y_pred_proba: numpy.array, shape (number of examples,)
            The score or probability output by the model. The probability
            of y_true being 1 should increase as this value
            increases.    
    
    output_path: the location to save the plot. Default is None.    
    """      
    interval_index = pd.IntervalIndex(pd.qcut(pd.Series(y_pred_proba).sort_values(ascending=False), 10, duplicates='drop').drop_duplicates()) 
    group = pd.Series([interval_index.get_loc(element) for element in y_pred_proba])
    distribution = pd.DataFrame({
            'group':group,
            'y_true':y_true
            })
    grouped = distribution.groupby('group')
    pct_of_target = grouped['y_true'].sum() / np.sum(y_true)
    '''
    fig=plt.figure(figsize=(15,7))
    # 提升图
    ax1=fig.add_subplot(1,2,1)
    ax2=fig.add_subplot(1,2,2)
    ax1.bar(range(1,11,1), pct_of_target, width=0.4, label='模型')
    ax1.bar(np.arange(1,11,1)+0.4, [(np.sum(y_true)/len(y_true))]*10, width=0.4, label='随机')
    ax1.set_title('提升图', fontdict=font_title)
    ax1.set_xlabel('Y=1的概率的分组', fontdict=font_text)
    ax1.set_ylabel('各分组Y=1占全部Y=1的比例', fontdict=font_text)
    ax1.legend()
    # 累计提升图
    ax2.plot(range(0,11,1), [0]+list(pct_of_target.cumsum()),  label='模型')
    ax2.plot([0,10],[0,1],label='随机')
    ax2.set_title('累计提升图', fontdict=font_title)
    ax2.set_xlabel('Y=1的概率的分组', fontdict=font_text)
    ax2.set_ylabel('累计Y=1占全部Y=1的比例', fontdict=font_text)
    ax2.set_xlim((0,10))
    ax2.set_ylim((0,1))
    ax2.legend()
    '''
    fig=plt.figure(figsize=(15,7))
    # 提升图
    ax1=fig.add_subplot(1,2,1)
    ax2=fig.add_subplot(1,2,2)
    ax1.bar(range(1,len(pct_of_target)+1,1), pct_of_target, width=0.4, label='模型')
    ax1.bar(np.arange(1,len(pct_of_target)+1,1)+0.4, [(np.sum(y_true)/len(y_true))]*len(pct_of_target), width=0.4, label='随机')
    ax1.set_title('提升图', fontdict=font_title)
    ax1.set_xlabel('Y=1的概率的分组', fontdict=font_text)
    ax1.set_ylabel('各分组Y=1占全部Y=1的比例', fontdict=font_text)
    ax1.legend()
    # 累计提升图
    ax2.plot(range(0,len(pct_of_target)+1,1), [0]+list(pct_of_target.cumsum()),  label='模型')
    ax2.plot([0,len(pct_of_target)],[0,1],label='随机')
    ax2.set_title('累计提升图', fontdict=font_title)
    ax2.set_xlabel('Y=1的概率的分组', fontdict=font_text)
    ax2.set_ylabel('累计Y=1占全部Y=1的比例', fontdict=font_text)
    ax2.set_xlim((0,len(pct_of_target)))
    ax2.set_ylim((0,1))
    ax2.legend()
    
    if output_path is not None:
        plt.savefig(output_path+r'提升图.png',dpi=500,bbox_inches='tight')
    plt.show()

def plot_all(y_true, y_pred_proba, output_path=None):
    """Output all plots to evaluate binary classification.
    
    Parameters
    ----------
    y_true: numpy.array, shape (number of examples,)
            The target column (or dependent variable).  
    
    y_pred_proba: numpy.array, shape (number of examples,)
            The score or probability output by the model. The probability
            of y_true being 1 should increase as this value
            increases.    
    
    output_path: the location to save the plot. Default is None.    
    """    
    roc(y_true, y_pred_proba, output_path=output_path)
    plt.close()
    ks(y_true, y_pred_proba, output_path=output_path)
    plt.close()
    lift_curve(y_true, y_pred_proba, output_path=output_path)
    plt.close()
    precision_recall(y_true, y_pred_proba, output_path=output_path)
    plt.close()



    
    