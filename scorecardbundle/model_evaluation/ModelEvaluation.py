# -*- coding: utf-8 -*-
"""
Model evaluation for binary classification task.

@authors: Lantian ZHANG
"""

import pandas as pd
import scipy
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve,average_precision_score
import numpy as np

# Global settings for matplotlib
plt.rcParams['font.sans-serif'] = ['SimHei'] # So that Chinese can be displayed
plt.rcParams['axes.unicode_minus'] = False # So that '-' can be displayed

plt.style.use('seaborn-colorblind') # Set style for matplotlib
plt.rcParams['savefig.dpi'] = 300 # dpi of diagrams
plt.rcParams['figure.dpi'] = 120

# Define fonts for texts in matplotlib
font_text = {'family':'SimHei',
        'weight':'normal',
         'size':12,
        } # font for notmal text

font_title = {'family':'SimHei',
        'weight':'bold',
         'size':16,
        } # font for title


# ============================================================
# Plot evaluation results
# ============================================================

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

            If Scorecard model's parameter "PDO" is negative, then the higher the 
            model scores, the higher the probability of y_pred being 1. This Function
            works fine. 

            However!!! if the parameter "PDO" is positive, then the higher 
            the model scores, the lower the probability of y_pred being 1. In this case,
            just put a negative sign before the scores array and pass `-scores` as parameter
            y_pred_proba of this function.              
    """
    ks = scipy.stats.ks_2samp(y_pred_proba[y_true==1], y_pred_proba[y_true!=1]).statistic
    return ks

def plot_ks(y_true, y_pred_proba, output_path=None):
    """Plot K-S curve of a model
    Parameters
    ----------
    y_true: numpy.array, shape (number of examples,)
            The target column (or dependent variable).  
    
    y_pred_proba: numpy.array, shape (number of examples,)
            The score or probability output by the model. The probability
            of y_true being 1 should increase as this value
            increases.

            If Scorecard model's parameter "PDO" is negative, then the higher the 
            model scores, the higher the probability of y_pred being 1. This Function
            works fine. 

            However!!! if the parameter "PDO" is positive, then the higher 
            the model scores, the lower the probability of y_pred being 1. In this case,
            just put a negative sign before the scores array and pass `-scores` as parameter
            y_pred_proba of this function. 
    
    output_path: string, optional(default=None)
        the location to save the plot. 
        e.g. r'D:\\Work\\jupyter\\'.
    """    
    # Check input data 
    if isinstance(y_true, pd.Series):
        target = y_true.values
    elif isinstance(y_true, np.ndarray):
        target = y_true
    else:
        raise TypeError('y_true should be either numpy.array or pandas.Series')

    if isinstance(y_pred_proba, pd.Series):
        scores = y_pred_proba.values
    elif isinstance(y_pred_proba, np.ndarray):
        scores = y_pred_proba
    else:
        raise TypeError('y_pred_proba should be either numpy.array or pandas.Series')

    # Group scores into 10 groups ascendingly
    interval_index = pd.IntervalIndex(pd.qcut(
        pd.Series(scores).sort_values(ascending=False), 10, duplicates='drop'
                                              ).drop_duplicates()) 
    group = pd.Series([interval_index.get_loc(element) for element in scores])

    distribution = pd.DataFrame({'group':group,
                                 'y_true':target
                                 })
    grouped = distribution.groupby('group')
    pct_of_target = grouped['y_true'].sum() / np.sum(target)
    pct_of_nontarget = (grouped['y_true'].size() - grouped['y_true'].sum()) / (len(target) - np.sum(target))
    cumpct_of_target = pd.Series([0] + list(pct_of_target.cumsum()))
    cumpct_of_nontarget = pd.Series([0] + list(pct_of_nontarget.cumsum()))
    diff = cumpct_of_target - cumpct_of_nontarget
    
    # Plot ks curve
    plt.plot(cumpct_of_target, label='Y=1')
    plt.plot(cumpct_of_nontarget, label='Y=0')
    plt.plot(diff, label='K-S curve')
    ks = round(diff.abs().max(),3)
    print('KS = '+str(ks))
    plt.annotate(s='KS = '+str(ks) ,xy=(diff.abs().idxmax(),diff.abs().max()))
    plt.xlim((0,10))
    plt.ylim((0,1))
    plt.title('K-S Curve', fontdict=font_title)   
    plt.xlabel('Group of scores', fontdict=font_text)
    plt.ylabel('Cumulated class proportion', 
                fontdict=font_text)    
    plt.legend()

    if output_path is not None:
        plt.savefig(output_path+r'K-S_Curve.png', dpi=500, bbox_inches='tight')    
    plt.show()
        
# ROC curve
def plot_roc(y_true, y_pred_proba, output_path=None):
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

            If Scorecard model's parameter "PDO" is negative, then the higher the 
            model scores, the higher the probability of y_pred being 1. This Function
            works fine. 

            However!!! if the parameter "PDO" is positive, then the higher 
            the model scores, the lower the probability of y_pred being 1. In this case,
            just put a negative sign before the scores array and pass `-scores` as parameter
            y_pred_proba of this function.    
    
    output_path: string, optional(default=None)
        the location to save the plot. 
        e.g. r'D:\\Work\\jupyter\\'.
    """    
    # Check input data 
    if isinstance(y_true, pd.Series):
        target = y_true.values
    elif isinstance(y_true, np.ndarray):
        target = y_true
    else:
        raise TypeError('y_true should be either numpy.array or pandas.Series')

    if isinstance(y_pred_proba, pd.Series):
        scores = y_pred_proba.values
    elif isinstance(y_pred_proba, np.ndarray):
        scores = y_pred_proba
    else:
        raise TypeError('y_pred_proba should be either numpy.array or pandas.Series')

    # Plot
    print('AUC:',roc_auc_score(target, scores)) #AUC
    fpr, tpr, thresholds = roc_curve(target, scores)
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate',fontdict=font_text)
    plt.ylabel('True Positive Rate',fontdict=font_text)
    plt.annotate(s='AUC = '+str(round(roc_auc_score(target, scores),3)), 
                 xy = (0.03, 0.95))
    plt.title('ROC Curve', fontdict=font_title)

    if output_path is not None:
        plt.savefig(output_path+r'ROC_Curve.png',dpi=500,bbox_inches='tight')
    plt.show()

# Precision vs Recall
def plot_precision_recall(y_true, y_pred_proba, output_path=None):
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

            If Scorecard model's parameter "PDO" is negative, then the higher the 
            model scores, the higher the probability of y_pred being 1. This Function
            works fine. 

            However!!! if the parameter "PDO" is positive, then the higher 
            the model scores, the lower the probability of y_pred being 1. In this case,
            just put a negative sign before the scores array and pass `-scores` as parameter
            y_pred_proba of this function.    
    
    output_path: string, optional(default=None)
        the location to save the plot. 
        e.g. r'D:\\Work\\jupyter\\'.
    """    
    # Check input data 
    if isinstance(y_true, pd.Series):
        target = y_true.values
    elif isinstance(y_true, np.ndarray):
        target = y_true
    else:
        raise TypeError('y_true should be either numpy.array or pandas.Series')

    if isinstance(y_pred_proba, pd.Series):
        scores = y_pred_proba.values
    elif isinstance(y_pred_proba, np.ndarray):
        scores = y_pred_proba
    else:
        raise TypeError('y_pred_proba should be either numpy.array or pandas.Series')

    precisions, recalls, thresholds = precision_recall_curve(target, scores)
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold', fontdict=font_text)
    plt.ylabel('Precision/Recall score', fontdict=font_text)
    plt.legend(loc='center left')
    plt.title('Precision vs Recall Curve', fontdict=font_title)

    if output_path is not None:
        plt.savefig(output_path+r'Precision_Recall_Curve.png',dpi=500,bbox_inches='tight')
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

            If Scorecard model's parameter "PDO" is negative, then the higher the 
            model scores, the higher the probability of y_pred being 1. This Function
            works fine. 

            However!!! if the parameter "PDO" is positive, then the higher 
            the model scores, the lower the probability of y_pred being 1. In this case,
            just put a negative sign before the scores array and pass `-scores` as parameter
            y_pred_proba of this function.    
    
    output_path: the location to save the plot. Default is None.    
    """    
    # Check input data 
    if isinstance(y_true, pd.Series):
        target = y_true.values
    elif isinstance(y_true, np.ndarray):
        target = y_true
    else:
        raise TypeError('y_true should be either numpy.array or pandas.Series')

    if isinstance(y_pred_proba, pd.Series):
        scores = y_pred_proba.values
    elif isinstance(y_pred_proba, np.ndarray):
        scores = y_pred_proba
    else:
        raise TypeError('y_pred_proba should be either numpy.array or pandas.Series')

    plot_ks(target, scores, output_path=output_path)
    plt.close()
    plot_roc(target, scores, output_path=output_path)
    plt.close()
    plot_precision_recall(target, scores, output_path=output_path)
    plt.close()

class BinaryTargets():
    """Model evaluation for binary classification problem.
    
    Parameters
    ----------
    y_true: numpy.array, shape (number of examples,)
            The target column (or dependent variable).  
    
    y_pred_proba: numpy.array, shape (number of examples,)
            The score or probability output by the model. The probability
            of y_true being 1 should increase as this value
            increases.

            If Scorecard model's parameter "PDO" is negative, then the higher the 
            model scores, the higher the probability of y_pred being 1. This Function
            works fine. 

            However!!! if the parameter "PDO" is positive, then the higher 
            the model scores, the lower the probability of y_pred being 1. In this case,
            just put a negative sign before the scores array and pass `-scores` as parameter
            y_pred_proba of this function.    
    
    output_path: string, optional(default=None)
            the location to save the plot, e.g. r'D:\\Work\\jupyter\\'.

    Methods
    -------
    ks_stat(): Return the k-s stat
    plot_ks(): Draw k-s curve
    plot_roc(): Draw ROC curve
    plot_precision_recall(): Draw precision recall curve
    plot_all(): Draw k-s, ROC curve, and precision recall curve
    """   
    def __init__(self, y_true, y_pred_proba=None, y_pred=None, output_path=None):

        self.__output_path__ = output_path

        if isinstance(y_true, pd.Series):
            self.__y_true__ = y_true.values
        elif isinstance(y_true, np.ndarray):
            self.__y_true__ = y_true
        elif y_true is None:
            self.__y_true__ = None
        else:
            raise TypeError('y_true should be either numpy.array or pandas.Series')

        if isinstance(y_pred_proba, pd.Series):
            self.__y_pred_proba__ = y_pred_proba.values
        elif isinstance(y_pred_proba, np.ndarray):
            self.__y_pred_proba__ = y_pred_proba
        elif y_pred_proba is None:
            self.__y_pred_proba__ = None
        else:
            raise TypeError('y_pred_proba should be either numpy.array or pandas.Series')

        if isinstance(y_pred, pd.Series):
            self.__y_pred__ = y_pred.values
        elif isinstance(y_pred, np.ndarray):
            self.__y_pred__ = y_pred
        elif y_pred is None:
            self.__y_pred__ = None
        else:
            raise TypeError('y_pred should be either numpy.array or pandas.Series')
                                  
    def ks_stat(self):
        return ks_stat(self.__y_true__, self.__y_pred_proba__)
    
    def plot_ks(self):
        return plot_ks(self.__y_true__, self.__y_pred_proba__, 
                        output_path=self.__output_path__)
    
    def plot_roc(self):
        return plot_roc(self.__y_true__, self.__y_pred_proba__, 
                        output_path=self.__output_path__)
    
    def plot_precision_recall(self):
        return plot_precision_recall(self.__y_true__, self.__y_pred_proba__, 
                        output_path=self.__output_path__)
    
    def plot_all(self):
        return plot_all(self.__y_true__, self.__y_pred_proba__, 
                        output_path=self.__output_path__)


# ============================================================
# Classification performance table
# Thoroughly evaluate model's ranking power over the given event
# ============================================================

def pref_table(y_true,y_pred_proba,thresholds=None,rename_dict={}):
    """Evaluate the classification performance on differet levels of model scores (y_pred_proba).
    Useful for setting classification threshold based on requirements of precision and recall.

    Parameters
    ----------
    y_true: numpy.array, shape (number of examples,)
            The target column (or dependent variable).  
    
    y_pred_proba: numpy.array, shape (number of examples,)
            The score or probability output by the model. The probability
            of y_true being 1 should increase as this value
            increases.

            If Scorecard model's parameter "PDO" is negative, then the higher the 
            model scores, the higher the probability of y_pred being 1. This Function
            works fine. 

            However!!! if the parameter "PDO" is positive, then the higher 
            the model scores, the lower the probability of y_pred being 1. In this case,
            just put a negative sign before the scores array and pass `-scores` as parameter
            y_pred_proba of this function.   
    
    thresholds: iterable. Can be list, numpy.array, etc.
            The thresholds used to turn model scores into groups so that each group's
            performance can be evaluated.
    
    rename_dict: python dictionary.
            A dictionary that maps the column names of the returned table to user-defined names.
            Use this parameter to change the name of the returned table.
            For example, inputing {'cum_f1':'cumulated_f1_score'} would rename the column 'cum_f1'
            of the returned table as 'cumulated_f1_score'
    """
    # Print AUC and AP
    print(f'roc_auc_score={roc_auc_score(y_true, y_pred_proba)}')
    print(f'AP={average_precision_score(y_true, y_pred_proba)}')

    # Result dataframe
    res = pd.DataFrame({
        'y_true':y_true
        ,'y_pred_proba':y_pred_proba
    })

    # Define the thresholds to bin model scores into different groups
    if thresholds is None: # Default thresholds
        thresholds = [-float('inf')]+list(np.concatenate([np.arange(1,10,1)/1000,np.arange(1,100,1)/100],axis=0))+[float('inf')]
    else: # User-defined thresholds
        thresholds = sorted(list(set([-float('inf')]+list(thresholds)+[float('inf')])))
    res['y_pred_group'] = pd.cut(res['y_pred_proba'].values,thresholds)   
    
    # Classification performance on each score interval
    stat = res.groupby('y_pred_group')['y_true'].sum().reset_index().rename(columns={'y_true':'event_num'})
    stat['sample_size']  = res.groupby('y_pred_group')['y_true'].size().values
    stat.sort_values('y_pred_group',ascending=False,inplace=True) # The higher the scores, the higher the probability of y_pred being 1
    stat['cum_event_num'] = stat['event_num'].cumsum()
    stat['cum_sample_size'] = stat['sample_size'].cumsum()
    stat['cum_sample_pct'] = stat['cum_sample_size']/stat['sample_size'].sum()
    stat['cum_precision'] = stat['cum_event_num']/stat['cum_sample_size']
    stat['cum_recal'] = stat['cum_event_num']/stat['event_num'].sum()
    stat['cum_f1'] = 2/(1/stat['cum_precision']+1/stat['cum_recal'])
    
    return stat.rename(columns=rename_dict) # Allow renameing the output table

