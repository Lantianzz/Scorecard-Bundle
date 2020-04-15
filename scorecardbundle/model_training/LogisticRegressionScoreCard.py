# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9  12:03:43 2018
Updated on Thu Dec  13 15:35:00 2018

@author: Lantian ZHANG <peter.lantian.zhang@outlook.com>

Python module for standard scorecard modeling 
"""

from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ============================================================
# Basic Functions
# ============================================================

def _assign_interval_base(x, boundaries):
    """Assign each value in x an interval from boundaries.
    
    Parameters
    ----------
    x: numpy.array, shape (number of examples,)
        The column of data that need to be discretized.
    
    boundaries: numpy.array, shape (number of interval boundaries,)
        The boundary values of the intervals to discretize target x. 
    
    Returns
    -------
    intervals: numpy.ndarray, shape (number of examples,2)
        The array of intervals that are closed to the right. 
        The left column and right column of the array are the 
        left and right boundary respectively.
    """    
    # Add -inf and inf to the start and end of boundaries
    boundaries = np.unique(
        np.concatenate((np.array([-float('inf')]),
                        boundaries,np.array([float('inf')])), 
                        axis=0))
    # The max boundary that is smaller than x_i is its lower boundary.
    # The min boundary that is >= than x_i is its upper boundary. 
    # Adding equal is because all intervals here are closed to the right.
    boundaries_diff_boolean = x.reshape(1,-1).T > boundaries.reshape(1,-1) 
    lowers = [boundaries[b].max() for b in boundaries_diff_boolean] 
    uppers = [boundaries[b].min() for b in ~boundaries_diff_boolean] 
    # Array of intervals that are closed to the right
    intervals= np.stack((lowers, uppers), axis=1) 
    return intervals

def assign_interval_str(x, boundaries, delimiter='~'):
    """Assign each value in x an interval from boundaries.
    
    Parameters
    ----------
    x: numpy.array, shape (number of examples,)
        The column of data that need to be discretized.
    
    boundaries: numpy.array, shape (number of interval boundaries,)
        The boundary values of the intervals to discretize target x. 

    delimiter: string, optional(default='~')
        The returned array will be an array of intervals. Each interval is 
        representated by string (i.e. '1~2'), which takes the form 
        lower+delimiter+upper. This parameter control the symbol that 
        connects the lower and upper boundaries.
    Returns
    -------
    intervals_str: numpy.array, shape (number of examples,)
        Discretized result. The array of intervals that represented by strings. 
    """
    intervals= _assign_interval_base(x, boundaries)
    # use join rather than use a+delimiter+b makes this line faster
    intervals_str = np.array(
        [delimiter.join((str(a),str(b))) for a,b in zip(intervals[:,0],
                                                        intervals[:,1])]
        )
    return intervals_str

def interval_to_boundary_vector(vector, delimiter='~'):
    """Transform an array of interval strings into the 
    unique boundaries of such intervals.
    
    Parameters
    ----------
    vector: numpy.array, shape (number of examples,)
        The array of interval whose unique boundaries will 
        be returned.

    delimiter: string, optional(default='~')
        The interval is representated by string (i.e. '1~2'), 
        which takes the form lower+delimiter+upper. This parameter 
        control the symbol that connects the lower and upper boundaries.    
    Returns
    -------
    boundaries: numpy.array, shape (number of interval boundaries,)
        An array of boundary values.
    """
    boundaries = np.array(list(set(delimiter.join(np.unique(vector)).split(delimiter))))
    boundaries = boundaries[(boundaries!='-inf') & (boundaries!='inf')].astype(float)
    return boundaries

def map_np(array, dictionary):
    """map function for numpy array
    Parameters
    ----------
    array: numpy.array, shape (number of examples,)
            The array of data to map values to.
    
    distionary: dict
            The distionary object.

    Return
    ----------
    result: numpy.array, shape (number of examples,)
            The mapped result.         
    """
    return [dictionary[e] for e in array]

def _applyScoreCard(scorecard, feature_name, feature_array, delimiter='~'):
    """Apply the scorecard to a column. Return the score for each value.
    
    Parameters
    ----------
    scorecard: pandas.DataFrame,
        the Scorecard rule table.   
    
    feature_name: string, 
        the name of the feature to score.
    
    feature_array: numpy.array, 
        the values of the feature to score.
    """
    score_rules = scorecard[scorecard['feature']==feature_name]
    boundaries = interval_to_boundary_vector(score_rules.value.values, delimiter=delimiter)
    intervals = assign_interval_str(feature_array, boundaries, delimiter=delimiter)
    score_dict = dict(zip(score_rules.value, score_rules.score))
    scores = map_np(intervals, score_dict)
    return scores

# ============================================================
# Main Functions
# ============================================================

class LogisticRegressionScoreCard(BaseEstimator, TransformerMixin):
    """Take woe-ed features, fit a regression and turn it into a scorecard.
    
    Parameters
    ----------
    woe_transformer: WOE transformer object from WOE module.

    C:  float, optional(Default=1.0)
        regularization parameter in linear regression. Default value is 1. 
        A smaller value implies more regularization.
        See details in scikit-learn document.

    class_weight: dict, optional(default=None)
        weights for each class of samples (e.g. {class_label: weight}) 
        in linear regression. This is to deal with imbalanced training data. 
        Setting this parameter to 'auto' will aotumatically use 
        class_weight function from scikit-learn to calculate the weights. 
        The equivalent codes are:
        >>> from sklearn.utils import class_weight
        >>> class_weights = class_weight.compute_class_weight('balanced', 
                                                              np.unique(y), y)

    random_state: int, optional(default=None)
        random seed in linear regression. See details in scikit-learn doc.

    PDO: int,  optional(default=-20)
        Points to double odds. One of the parameters of Scorecard.
        Default value is -20. 
        A positive value means the higher the scores, the lower 
        the probability of y being 1. 
        A negative value means the higher the scores, the higher 
        the probability of y being 1.

    basePoints: int,  optional(default=100)
        the score for base odds(# of y=1/ # of y=0).

    decimal: int,  optional(default=0)
        Control the number of decimals that the output scores have.
        Default is 0 (no decimal).

    start_points: boolean, optional(default=False)
        There are two types of scorecards, with and without start points.
        True means the scorecard will have a start poitns. 

    output_option: string, optional(default='excel')
        Controls the output format of scorecard. For now 'excel' is 
        the only option.

    output_path: string, optional(default=None)
        The location to save the scorecard. e.g. r'D:\\Work\\jupyter\\'.

    verbose: boolean, optioanl(default=False)
        When verbose is set to False, the predict() method only returns
        the total scores of samples. In this case the output of predict() 
        method will be numpy.array;
        When verbose is set to True, the predict() method will return
        the total scores, as well as the scores of each feature. In this case
        The output of predict() method will be pandas.DataFrame in order to 
        specify the feature names.

    delimiter: string, optional(default='~')
        The feature interval is representated by string (i.e. '1~2'), 
        which takes the form lower+delimiter+upper. This parameter 
        is the symbol that connects the lower and upper boundaries.
    
    Attributes
    ---------- 
    woe_df_: pandas.DataFrame, the scorecard.
    
    AB_ : A and B when converting regression to scorecard.

    Methods
    -------
    fit(woed_X, y): 
            fit the Scorecard model.

    predict(X_beforeWOE, load_scorecard=None): 
            Apply the model to the original feature 
            (before discretization and woe encoding).
            If user choose to upload their own Scorecard,
            user can pass a pandas.DataFrame to `load_scorecard`
            parameter. The dataframe should contain columns such as 
            feature, value, woe, beta and score. An example would
            be as followed (value is the range of feature values, woe 
            is the WOE encoding of that range, and score is the socre
            for that range):
            feature value   woe         beta        score
            x1      30~inf  0.377563    0.631033    5.0
            x1      20~-30  1.351546    0.631033    37.0
            x1      -inf~20 1.629890    0.631033    -17.0
    
    """     
    
    def __init__(self, woe_transformer, C=1, class_weight=None, random_state=None,
                 PDO=-20, basePoints = 100, decimal=0, start_points = False,
                 output_option='excel', output_path=None, verbose=False, 
                 delimiter='~'):
        
        self.__woe_transformer__ = woe_transformer
        self.__C__ = C
        self.__class_weight__ = class_weight
        self.__random_state__ = random_state
        self.__PDO__ = PDO
        self.__basePoints__ = basePoints
        self.__output_option__ = output_option
        self.__decimal__ = decimal
        self.__output_path__ = output_path
        self.__start_points__ = start_points
        self.__verbose__ = verbose
        self.__delimiter__ = delimiter
    
    def fit(self, woed_X, y):
        """
        Parameters
        ----------
        woed_X: numpy.ndarray or pandas.DataFrame, shape (number of examples, 
                                                     number of features)
            The woe encoded X.
        
        y: numpy.array or pandas.Series, shape (number of examples,)
            The target array (or dependent variable).
        """ 

        # if X is pandas.DataFrame, turn it into numpy.ndarray and 
        # associate each column array with column names.
        # if X is numpy.ndarray, generate column names for it (x1, x2,...)
        self.fit_sample_size_, self.num_of_x_ = woed_X.shape
        if isinstance(woed_X, pd.DataFrame):
            self.columns_ = woed_X.columns.values # column names
            features = woed_X.values
        elif isinstance(woed_X, np.ndarray):
            self.columns_ = np.array(
                [''.join(('x',str(a))) for a in range(self.num_of_x_)]
                ) #  # column names (i.e. x0, x1, ...)
            features = woed_X
        else:
            raise TypeError('woed_X should be either numpy.ndarray or pandas.DataFrame')

        if isinstance(y, pd.Series):
            target = y.values
        elif isinstance(y, np.ndarray):
            target = y
        else:
            raise TypeError('y should be either numpy.array or pandas.Series')

        # Basic settings of Scorecard
        positive, total = y.sum(), y.shape[0]   
        self.__baseOdds__ = positive / (total - positive) 
        self.__p__ = self.__baseOdds__  / (1 + self.__baseOdds__)
        B = self.__PDO__/np.log(2)
        A = self.__basePoints__ + B * np.log(self.__baseOdds__)
        self.AB_ = (A, B)
        
        # Concat vertically all woe values (scorecard rules table)
        woe_list = [pd.concat([
           pd.Series([col]*len(self.__woe_transformer__.result_dict_[col][0])), # Name of x
           pd.Series(list(self.__woe_transformer__.result_dict_[col][0].keys())), # interval strings of x 
           pd.Series(list(self.__woe_transformer__.result_dict_[col][0].values())) # woe values of x
           ],axis=1) for col in self.columns_]
        self.woe_df_ = pd.concat(woe_list, axis=0, ignore_index=True
                                 ).rename(columns={0:'feature',1:'value',2:'woe'})

        # Fit a logistic regression
        lr = LogisticRegression(C=self.__C__, 
                                class_weight=self.__class_weight__,
                                random_state=self.__random_state__)
        lr.fit(features, y)
        
        # Calculate scores for each value in each feature, and the start scores
        beta_map = dict(zip(list(self.columns_), 
                            lr.coef_[0,:]))
        self.woe_df_['beta'] = map_np(self.woe_df_.feature, beta_map)
        self.startPoints_ = A - B * lr.intercept_[0]    
        
        # Rule table for Scoracard
        if self.__start_points__ is True:
            self.woe_df_['score'] = np.around(
                -B * self.woe_df_['beta'].values * self.woe_df_['woe'].values, 
                decimals=self.__decimal__)
            startPoints = pd.DataFrame({'feature': ['StartPoints'],
                'value': [np.nan],
                'woe': [np.nan],
                'beta': [np.nan],
                'score': np.around(self.startPoints_, decimals=self.__decimal__)
                })
            #the scorecard
            self.woe_df_ = pd.concat([startPoints, self.woe_df_],
                                     axis=0,
                                     ignore_index=True)  
        elif self.__start_points__ is False:  
            self.woe_df_['score'] = np.around(
                -B * self.woe_df_['beta'].values * self.woe_df_['woe'].values + self.startPoints_ / self.num_of_x_, 
                decimals=self.__decimal__)

        # Output the scorecard
        if self.__output_option__ == 'excel' and self.__output_path__ is None:
            self.woe_df_.to_excel('scorecards.xlsx', index=False)
        elif self.__output_option__ == 'excel' and self.__output_path__ is not None:
            self.woe_df_.to_excel(self.__output_path__+'scorecards.xlsx', index=False)
        self.lr_ = lr

    def predict(self, X_beforeWOE, load_scorecard=None):
        """Apply the scorecard.
        
        Parameters
        ----------
        X_beforeWOE: numpy.ndarray or pandas.DataFrame, shape (number of examples, 
                                                     number of features)
            The features before WOE transformation (the original X).

        load_scorecard: pandas.DataFrame, optional(default=None)
            If we want to use a modified scorecard
            rather than the one automatically generated, we can pass the scorecard
            we want to use using this parameter. 
        """        

        # if X is pandas.DataFrame, turn it into numpy.ndarray and 
        # associate each column array with column names.
        # if X is numpy.ndarray, generate column names for it (x1, x2,...)
        self.transform_sample_size_ = X_beforeWOE.shape[0]
        if isinstance(X_beforeWOE, pd.DataFrame):
            features = X_beforeWOE.values.T
        elif isinstance(X_beforeWOE, np.ndarray):
            features = X_beforeWOE.T
        else:
            raise TypeError('X_beforeWOE should be either numpy.ndarray or pandas.DataFrame')

        # Check whether the user choose to load a Scorecard rule table
        if load_scorecard is None:
            scorecard = self.woe_df_
        else:
            scorecard = load_scorecard               

        # Apply the Scorecard rules
        scored_result = pd.concat(
            [pd.Series(_applyScoreCard(scorecard, 
                                       name,
                                       x, 
                                       delimiter=self.__delimiter__
                                        ), 
                        name=name
                        ) for name,x in zip(self.columns_, features)], 
            axis=1)
        scored_result['TotalScore'] = scored_result.sum(axis=1)
        if self.__verbose__:
            return scored_result
        else:
            return scored_result['TotalScore'].values
        