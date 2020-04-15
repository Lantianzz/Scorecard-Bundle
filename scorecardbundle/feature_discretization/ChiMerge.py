# -*- coding: utf-8 -*-
"""
Created on Thu Nov 1 2018
Updated on Sat Aug 10 2019

@authors: Lantian ZHANG <peter.lantian.zhang@outlook.com>

ChiMerge is a discretization algorithm introduced by Randy Kerber in 
"ChiMerge: Discretization of Numeric Attributes". It can transform 
a numerical features into categorical feature or reduce the number 
of intervals in a ordinal feature based on the feature's distribution 
and the target classes' relative frequencies in each interval. 
As a result, it keep statistically significantly different intervals 
and merge similar ones.
"""

import pandas as pd
import numpy as np
from scipy.stats import chi2
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
    max_value = max(x)
    boundaries = np.unique(
        np.concatenate((np.array([-float('inf')]),
                        boundaries,np.array([float('inf')])), 
                        axis=0))
    # The max boundary that is smaller than x_i is its lower boundary.
    # The min boundary that is >= than x_i is its upper boundary. 
    # Adding equal is because all intervals here are closed to the right.
    boundaries_diff_boolean = x.reshape(1,-1).T > boundaries.reshape(1,-1) 
    lowers = np.array([boundaries[b].max() for b in boundaries_diff_boolean])
    uppers = np.array([boundaries[b].min() for b in ~boundaries_diff_boolean])
    
    # Replace the upper value with inf if the value equals the maximum feature value
    n = x.shape[0]
    uppers = np.where(uppers==max_value,[float('inf')]*n,uppers) 
    # Array of intervals that are closed to the right
    intervals= np.stack((lowers, uppers), axis=1) 
    return intervals

def assign_interval_unique(x, boundaries):
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
    
    unique_intervals: numpy.ndarray, shape (number of unique intervals,2)
        The unique intervals that are closed to the right. 
        The left column and right column of the array are the 
        left and right boundary respectively.  
    """
    intervals= _assign_interval_base(x, boundaries)
    unique_intervals = np.unique(intervals, axis=0)
    return intervals, unique_intervals

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

def pivot_table_np(index, column):
    """Perform cross-tabulation to index vector and column vector.
    The returned pivot table has unique index values as its rows, 
    unique column values as its columns, and sample frequencies 
    of index-column combinations as its values.
    
    Parameters
    ----------
    index: numpy.array, shape (number of examples,)
             The vector whose unique values will be the rows of pivot table
    
    column: numpy.array, shape (number of examples,)
            The vector whose unique values will be the columns of pivot table
    
    Returns
    -------
    pivot_table: numpy.array, shape (number of unqiue index values, 
                                     number of unqiue column values)
             The pivot table that has unique index values as its rows, 
             unique column values as its columns, and sample frequencies 
             of index-column combinations as its values.
             
    column_unique: numpy.array, shape (number of unqiue column values,)
            Column labels of the pivot table. 
            The unqiue column values are sorted ascendingly.
    
    index_unique: numpy.array, shape (number of unqiue index values,)
            Row labels of the pivot table. 
            The unqiue index values are sorted ascendingly.
    """   

    # unique values of index and column
    column_unique, index_unique = np.unique(column), np.unique(index)
    n_j = column_unique.shape[0] # number of columns, number of indexes (rows)

    # Sample frequencies of each combination of index vector and column vector
    # Consider the value array has two hierarchical indexes (index & column)
    dot_multiply = np.stack((index, column), axis=1)
    dot_multiply, value = np.unique(dot_multiply, axis=0, return_counts=True) 

    # generate pivot table, there may be duplicated rows for index vector
    zeros = np.zeros(dot_multiply.shape[0])
    pivot_table = np.vstack(
        [np.where(dot_multiply[:,1]==j, value, zeros) for j in column_unique]
        ).T 
    
    # merge the pivot table rows with the same index value
    lowers = dot_multiply[:,0]
    pivot_table = np.vstack(
        [pivot_table[lowers==i,:].sum(axis=0) for i in index_unique]
        )
    return pivot_table, column_unique, index_unique

def chi2_test(A):
    """Calculate chi-square value to test whether the frequencies of 
    target classes are significantly differenrt in given intervals.
    Reference: "ChiMerge: DIscretization of Numerical Attributes"
    
    Parameters
    ----------
    A: numpy.ndarray, shape (number of invervals, number of classes)
       The frequency table that counts the number of examples in each
       interval and class (pivot table).
    
    Returns
    -------
    chi2: float
        The result of Chi square test
    """
    R = A.sum(axis=1).reshape(1,-1) # number of examples in each interval 
    C = A.sum(axis=0).reshape(1,-1).T # number of examples in each class
    # expected frequency
    # add 0.5 to avoid dividing by a small number (stated in original paper)
    E = np.dot(C, R) / C.sum()  + 0.5 
    chi2 = (np.square(A.T - E)/E).sum() # chi square value
    return chi2

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


# ============================================================
# Main Part
# ============================================================
def chi_merge_vector(x, y, m=2, confidence_level=0.9, max_intervals=None, 
                     min_intervals=1, initial_intervals=100, 
                     delimiter='~', output_boundary=False):
    """Merge similar adjacent m intervals until all adjacent 
    intervals are significantly different from each other.
    
    Parameters
    ----------

    x: numpy.array, shape (number of examples,)
        The array of data that need to be discretized.
    
    y: numpy.array, shape (number of examples,)
        The target array (or dependent variable).
    
    m: integer, optional(default=2)
        The number of adjacent intervals to compare during chi-squared test.
    
    confidence_level: float, optional(default=0.9)
        The confidence level to determine the threshold for intervals to 
        be considered as different during the chi-square test.
    
    max_intervals: int, optional(default=None)
        Specify the maximum number of intervals the discretized array will have.
        Sometimes (like when training a scorecard model) fewer intervals are 
        prefered. If do not need this option just set it to None.

    min_intervals: int, optional(default=1)
        Specify the mininum number of intervals the discretized array will have.
        If do not need this option just set it to 1.

    initial_intervals: int, optional(default=100)
        The original Chimerge algorithm starts by putting each unique value 
        in an interval and merging through a loop. This can be time-consumming 
        when sample size is large. 
        Set the initial_intervals option to values other than None (like 10 or 100) 
        will make the algorithm start at the number of intervals specified (the 
        initial intervals are generated using quantiles). This can greatly shorten 
        the run time. If do not need this option just set it to None.
     
    delimiter: string, optional(default='~')
        The returned array will be an array of intervals. Each interval is 
        representated by string (i.e. '1~2'), which takes the form 
        lower+delimiter+upper. This parameter control the symbol that 
        connects the lower and upper boundaries.
    
    output_boundary: boolean, optional(default=False)
        If output_boundary is set to True. This function will output the 
        unique upper  boundaries of discretized array. If it is set to False,
        This funciton will output the discretized array.
        For example, if it is set to True and the array is discretized into 
        3 groups (1,2),(2,3),(3,4), this funciton will output an array of 
        [1,3,4].

    Return
    ------
    intervals_str: numpy.array, shape (number of examples,)
        Discretized result. The array of intervals that represented by strings.
    """
    # Initialization step

    n_j = np.unique(y).shape[0] # number of classes
    n_i = np.unique(x).shape[0] # number of unique x values
    if (initial_intervals is not None and 
        initial_intervals < n_i and 
        n_i > min_intervals):
        # Use quantiles to bin x
        boundaries = np.unique(
            np.quantile(x, np.arange(0, 1, 1/initial_intervals)[1:])
            ) # Add [1:] so that 0% persentile will not be a threshold
        intervals, unique_intervals = assign_interval_unique(x, boundaries)
    else:
        # Put each unique value of x in its own interval
        boundaries = np.unique(x)
        intervals, unique_intervals = assign_interval_unique(x, boundaries)     
    # Return unique values as result if the # of unique x <= min_intervals
    if n_i <= min_intervals and output_boundary is False: 
        intervals_str = np.array(
            [delimiter.join((str(a),str(b))) for a,b in zip(intervals[:,0],
                                                            intervals[:,1])])
        return intervals_str
    elif n_i <= min_intervals and output_boundary is True: 
        if len(np.unique(x))>1:
            boundaries = np.unique(
                np.concatenate((np.unique(x),
                                np.array([float('inf')])), 
                                axis=0))
        else:
            boundaries = np.array([float('inf')])
    
        return boundaries

    # Merging step

    if max_intervals is None:
        max_intervals = n_i
    threshold = chi2.ppf(confidence_level, n_j-1) # chi2 threshold
    # pivot table of index*column
    pt_value, pt_column, pt_index = pivot_table_np(intervals[:,1], y)

    # perform Chi-square test on each pair of m adjacent intervals 
    # use the ith interval's index as the interval pair's index    
    adjacent_list = (pt_value[i:i+m, :] for i in range(len(pt_value)-m+1)) 
    adjacent_index = np.array(
        [pt_index[i:i+m] for i in range(len(pt_value)-m+1)]
        ) 
    chi2_array = np.array([chi2_test(adj) for adj in adjacent_list]) 

    # if unique_intervals.shape[0] <= min_intervals, stop the loop
    # if min chi2 > threshold and # of unique_intervals<=max_intervals, 
    # stop the loop  
    while (((chi2_array.min() <= threshold) or (unique_intervals.shape[0] > max_intervals)) and 
           (unique_intervals.shape[0] > min_intervals)):
        # identify the index of adjacent pair(s) with smallest chi2 score 
        index_adjacent_to_merge, = np.where(chi2_array==chi2_array.min()) 
        # identify the interval (or intervals) with smallest chi2 score 
        i_merge = adjacent_index[index_adjacent_to_merge,:]

        # merge the intervals for each selected pair
        new_interval = np.array(
            [(unique_intervals[:,0][unique_intervals[:,1] == index[0]][0],
              index[1]) for index in i_merge])

        # delete selected intervals and add the merged intervals
        index_delete_merged = np.array(
            [np.where(unique_intervals[:,1]==e)[0][0] for e in i_merge.reshape(1,-1)[0]]
            )
        unique_intervals = np.vstack((
            np.delete(unique_intervals, index_delete_merged, axis=0), 
            new_interval
            )) 
        unique_intervals.sort(axis=0) 

        # Reassign intervals with the updated thresholds (unique_intervals)
        intervals, unique_intervals = assign_interval_unique(x, unique_intervals[:,1])
        pt_value, pt_column, pt_index = pivot_table_np(intervals[:,1], y) 
        # perform Chi-square test on each pair of m adjacent intervals 
        # use the ith interval's index as the interval pair's index
        adjacent_list = (pt_value[i:i+m, :] for i in range(len(pt_value)-m+1)) 
        adjacent_index = np.array([pt_index[i:i+m] for i in range(len(pt_value)-m+1)]) 
        chi2_array = np.array([chi2_test(adj) for adj in adjacent_list])
    
    if output_boundary:
        return unique_intervals[:,1]
    else:
        # use join rather than use a+delimiter+b makes this line faster
        intervals_str = np.array(
            [delimiter.join((str(a),str(b))) for a,b in zip(intervals[:,0],
                                                            intervals[:,1])]
            )    
        return intervals_str


class ChiMerge(BaseEstimator, TransformerMixin):
    """
    ChiMerge is a discretization algorithm introduced by Randy Kerber in 
    "ChiMerge: Discretization of Numeric Attributes". It can transform 
    numerical features into categorical features or reduce the number 
    of intervals in a ordinal feature based on the target classes' relative 
    frequencies in each interval. As a result, it keep statistically 
    significantly different intervals and merge similar ones.
    
    Parameters
    ----------

    m: integer, optional(default=2)
        The number of adjacent intervals to compare during chi-squared test.
    
    confidence_level: float, optional(default=0.9)
        The confidence level to determine the threshold for intervals to 
        be considered as different during the chi-square test.
    
    max_intervals: int, optional(default=None)
        Specify the maximum number of intervals the discretized array will have.
        Sometimes (like when training a scorecard model) fewer intervals are 
        prefered. If do not need this option just set it to None.

    min_intervals: int, optional(default=1)
        Specify the mininum number of intervals the discretized array will have.
        If do not need this option just set it to 1.

    initial_intervals: int, optional(default=100)
        The original Chimerge algorithm starts by putting each unique value 
        in an interval and merging through a loop. This can be time-consumming 
        when sample size is large. 
        Set the initial_intervals option to values other than None (like 10 or 100) 
        will make the algorithm start at the number of intervals specified (the 
        initial intervals are generated using quantiles). This can greatly shorten 
        the run time. If do not need this option just set it to None.
     
    delimiter: string, optional(default='~')
        The returned array will be an array of intervals. Each interval is 
        representated by string (i.e. '1~2'), which takes the form 
        lower+delimiter+upper. This parameter control the symbol that 
        connects the lower and upper boundaries.
    
    output_boundary: boolean, optional(default=False)
        If output_boundary is set to True. This function will output the 
        unique upper  boundaries of discretized array. If it is set to False,
        This funciton will output the discretized array.
        For example, if it is set to True and the array is discretized into 
        3 groups (1,2),(2,3),(3,4), this funciton will output an array of 
        [1,3,4].

    Attributes
    ----------
    boundaries_: dict
        A dictionary that maps feature name to its merged boundaries.
    fit_sample_size_: int
        The sampel size of fitted data.
    transform_sample_size_:  int
        The sampel size of transformed data.
    num_of_x_:  int
        The number of features.
    columns_:  iterable
        An array of list of feature names.

    Methods
    -------
    fit(X, y): 
        fit the ChiMerge algorithm to the feature.

    transform(X): 
        transform the feature using the ChiMerge fitted.

    fit_transform(X, y): 
        fit the ChiMerge algorithm to the feature and transform it.    
    """
   
    def __init__(self, m=2, confidence_level=0.9, max_intervals=None, 
                    min_intervals=1, initial_intervals=100, 
                    delimiter='~', output_dataframe=False):
        self.__m__ = m
        self.__confidence_level__ = confidence_level
        self.__max_intervals__ = max_intervals
        self.__min_intervals__ = min_intervals
        self.__initial_intervals__ = initial_intervals
        self.__delimiter__ = delimiter
        self.__output_dataframe__ = output_dataframe
    
    def fit(self, X, y):
        """
        Parameters
        ----------
        X: numpy.ndarray or pandas.DataFrame, shape (number of examples, 
                                                     number of features)
            The data that need to be discretized.
        
        y: numpy.array or pandas.Series, shape (number of examples,)
            The target array (or dependent variable).
        """ 

        # if X is pandas.DataFrame, turn it into numpy.ndarray and 
        # associate each column array with column names.
        # if X is numpy.ndarray, 
        self.fit_sample_size_, self.num_of_x_ = X.shape
        if isinstance(X, pd.DataFrame):
            self.columns_ = X.columns.values # column names
            features = X.values
        elif isinstance(X, np.ndarray):
            self.columns_ = np.array(
                [''.join(('x',str(a))) for a in range(self.num_of_x_)]
                ) #  # column names (i.e. x0, x1, ...)
            features = X
        else:
            raise TypeError('X should be either numpy.ndarray or pandas.DataFrame')

        if isinstance(y, pd.Series):
            target = y.values
        elif isinstance(y, np.ndarray):
            target = y
        else:
            raise TypeError('y should be either numpy.array or pandas.Series')

        boundary_list = [chi_merge_vector(
                            features[:,i], target
                            , m=self.__m__
                            , confidence_level=self.__confidence_level__
                            , max_intervals=self.__max_intervals__
                            , min_intervals=self.__min_intervals__
                            , initial_intervals=self.__initial_intervals__
                            , delimiter=self.__delimiter__
                            , output_boundary=True
                            ) for i in range(self.num_of_x_)]
        self.boundaries_ = dict(zip(self.columns_, boundary_list))
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X: numpy.ndarray or pandas.DataFrame, shape (number of examples, 
                                                     number of features)
            The data that need to be discretized.
        """ 

        # if X is pandas.DataFrame, turn it into numpy.ndarray and 
        # associate each column array with column names.
        # if X is numpy.ndarray, 
        self.transform_sample_size_ = X.shape[0]
        if isinstance(X, pd.DataFrame):
            features = X[self.columns_].values
        elif isinstance(X, np.ndarray):
            features = X
        else:
            raise TypeError('X should be either numpy.ndarray or pandas.DataFrame')

        result = np.array([assign_interval_str(
                                features[:,i],
                                self.boundaries_[col],
                                delimiter=self.__delimiter__
                                ) for i,col in enumerate(self.columns_)])

        if self.__output_dataframe__:
            return pd.DataFrame(result, index=self.columns_).T
        else:
            return result.T
