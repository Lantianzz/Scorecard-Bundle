# -*- coding: utf-8 -*-
"""
Created on Mon Dec 3 2018
Updated on Sat Aug 10 2019

@authors: Lantian ZHANG <zhanglantian1992@163.com>

The codes below perform WOE transformation and IV calculation for features 
The data format of features should be string or number.
"""
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def woe_vector(x, y, epslon=1e-10):
    """Calculate WOE and IV for 1 feature
    Parameters
    ----------
    x: numpy.array, shape (number of examples,)
            The column of data that need to woe-transformed.
            The data in col should be either string or number(integer/float)
    
    y: numpy.array, shape (number of examples,)
            The target column (or dependent variable).
            
    epslon: float, optional(default=1e-10)
            Replace 0 with a very small number during division 
            or logrithm to avoid infinite value.   
    
    Return
    ----------
    woe_dict: dict
            The dictionary that maps each unique value of x to woe values
    iv: float
            The information value for x with reference to y
    """
    
    # global goods/bads   
    positive, total = y.sum(), len(y)   
    negative = total - positive
    global_odds = positive / negative
    
    # goods, bads for each group
    x_unique, local_total = np.unique(x, return_counts=True)
    mask = (x.reshape(1,-1) == x_unique.reshape(1,-1).T) # identify groups
    local_pos = np.array([y[m].sum() for m in mask])
    local_neg = local_total - local_pos
    local_odds = local_pos/(local_neg+epslon)

    # woe and iv
    ratios = local_odds/(global_odds+epslon)
    woe_values = np.log(ratios+epslon)
    woe_dict = dict(zip(x_unique, woe_values))
    iv = np.sum((local_pos/positive - local_neg/negative) * woe_values)
    
    return woe_dict, iv

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

class WOE(BaseEstimator, TransformerMixin):
    """
    Perform WOE transformation for features and calculate the information
    value (IV) of features with reference to the target variable y.
    
    Parameters
    ----------
    epslon: float, optional(default=1e-10)
            Replace 0 with a very small number during division 
            or logrithm to avoid infinite value.       
    
    output_dataframe: boolean, optional(default=False)
            if output_dataframe is set to True. The transform() function will
            return pandas.DataFrame. If it is set to False, the output will
            be numpy ndarray.

    Attributes
    ----------
    iv_: a dictionary that contains feature names and their IV
    
    result_dict_: a dictionary that contains feature names and 
        their WOE result tuple. Each WOE result tuple contains the
        woe value dictionary and the iv for the feature.

    Methods
    -------
    fit(X, y): 
            fit the WOE transformation to the feature.

    transform(X): 
            transform the feature using the WOE fitted.

    fit_transform(X, y): 
            fit the WOE transformation to the feature and transform it.         
    """
    def __init__(self, epslon=1e-10, output_dataframe=False):
        self.__epslon__ = epslon
        self.__output_dataframe__ = output_dataframe

    def fit(self, X, y):
        """
        Parameters
        ----------
        X: numpy.ndarray or pandas.DataFrame, shape (number of examples, 
                                                     number of features)
            The data that need to be transformed.
        
        y: numpy.array or pandas.Series, shape (number of examples,)
            The target array (or dependent variable).
        """ 

        # if X is pandas.DataFrame, turn it into numpy.ndarray and 
        # associate each column array with column names.
        # if X is numpy.ndarray, 
        self.fit_sample_size_, self.num_of_x_ = X.shape
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns.values # column names
            features = X.values.T
        elif isinstance(X, np.ndarray):
            self.columns_ = np.array(
                [''.join(('x',str(a))) for a in range(self.num_of_x_)]
                ) #  # column names (i.e. x0, x1, ...)
            features = X.T
        else:
            raise TypeError('X should be either numpy.ndarray or pandas.DataFrame')

        if isinstance(y, pd.Series):
            target = y.values
        elif isinstance(y, np.ndarray):
            target = y
        else:
            raise TypeError('y should be either numpy.array or pandas.Series')

        # Perform woe transformation to each feature
        self.result_dict_ = dict(zip(self.columns_, 
                                    (woe_vector(x,y) for x in features)))
        # Extract iv from result
        self.iv_ = dict(zip(self.columns_, 
                            (self.result_dict_[x][1] for x in self.columns_)))
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X: numpy.ndarray or pandas.DataFrame, shape (number of examples, 
                                                     number of features)
            The data that need to be transformed.
        """ 
        # if X is pandas.DataFrame, turn it into numpy.ndarray and 
        # associate each column array with column names.
        # if X is numpy.ndarray, 
        self.transform_sample_size_ = X.shape[0]
        if isinstance(X, pd.DataFrame):
            features = X[self.columns_].values.T
        elif isinstance(X, np.ndarray):
            features = X.T
        else:
            raise TypeError('X should be either numpy.ndarray or pandas.DataFrame')

        # Apply fitted woe transformation to features
        result = np.array(
            [map_np(d, trans_woe.result_dict_[col][0]) for d,col in zip(features,
                                                                  self.columns_)])
        # Output
        if self.__output_dataframe__:
            return pd.DataFrame(result, index=self.columns_).T
        else:
            return result.T