# -*- coding: utf-8 -*-
"""
Numpy based functions shared among classes

@author: Lantian ZHANG
"""

import numpy as np

# ============================================================
# Basic Functions
# ============================================================


def _assign_interval_base(x, boundaries, force_inf=True):
    """Assign each value in x an interval from boundaries.

    Parameters
    ----------
    x: numpy.array, shape (number of examples,)
        The column of data that need to be discretized.

    boundaries: numpy.array, shape (number of interval boundaries,)
        The boundary values of the intervals to discretize target x.

    force_inf: bool. Whether to force the largest interval's
        right boundary to be positive infinity. Default is True.

        In the case when the upper boundary is not smaller then the maximum value,
        the largest interval output will be (xxx, upper].

        In tasks like fitting ChiMerge where the output intervals are supposed to
        cover the entire value space (-inf ~ inf), this parameter `force_inf`
        should be set to True so that the largest interval will be
        overwritten from (xxx, upper] to (xxx, inf]. In other words, the previous
        upper boundary value is abandoned.

        However when merely applying given boundaries, the output intervals should be
        exactly where the values belong according to the given boundaries and does not
        have to cover the entire value space. Users may only pass in a few values
        to transform into intervals, forcing the largest interval to have inf may generate
        intervals that did not exist.

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
                        boundaries, np.array([float('inf')])), axis=0))
    # The max boundary that is smaller than x_i is its lower boundary.
    # The min boundary that is >= than x_i is its upper boundary.
    # Adding equal is because all intervals here are closed to the right.
    boundaries_diff_boolean = x.reshape(1, -1).T > boundaries.reshape(1, -1)
    lowers = np.array([boundaries[b].max() for b in boundaries_diff_boolean])
    uppers = np.array([boundaries[b].min() for b in ~boundaries_diff_boolean])

    # If force_inf is True
    # Replace the upper value with inf if it is not smaller then the maximum
    # feature value
    n = x.shape[0]
    if force_inf:
        uppers = np.where(uppers >= max_value, [float('inf')] * n, uppers)
    # Array of intervals that are closed to the right
    intervals = np.stack((lowers, uppers), axis=1)
    return intervals


def assign_interval_unique(x, boundaries, force_inf=True):
    """Assign each value in x an interval from boundaries.

    Parameters
    ----------
    x: numpy.array, shape (number of examples,)
        The column of data that need to be discretized.

    boundaries: numpy.array, shape (number of interval boundaries,)
        The boundary values of the intervals to discretize target x.

    force_inf: bool. Whether to force the largest interval's
        right boundary to be positive infinity. Default is True.

        In the case when the upper boundary is not smaller then the maximum value,
        the largest interval output will be (xxx, upper].

        In tasks like fitting ChiMerge where the output intervals are supposed to
        cover the entire value space (-inf ~ inf), this parameter `force_inf`
        should be set to True so that the largest interval will be
        overwritten from (xxx, upper] to (xxx, inf]. In other words, the previous
        upper boundary value is abandoned.

        However when merely applying given boundaries, the output intervals should be
        exactly where the values belong according to the given boundaries and does not
        have to cover the entire value space. Users may only pass in a few values
        to transform into intervals, forcing the largest interval to have inf may generate
        intervals that did not exist.

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
    intervals = _assign_interval_base(x, boundaries, force_inf=force_inf)
    unique_intervals = np.unique(intervals, axis=0)
    return intervals, unique_intervals


def assign_interval_str(x, boundaries, delimiter='~', force_inf=True):
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

    force_inf: bool. Whether to force the largest interval's
        right boundary to be positive infinity. Default is True.

        In the case when the upper boundary is not smaller then the maximum value,
        the largest interval output will be (xxx, upper].

        In tasks like fitting ChiMerge where the output intervals are supposed to
        cover the entire value space (-inf ~ inf), this parameter `force_inf`
        should be set to True so that the largest interval will be
        overwritten from (xxx, upper] to (xxx, inf]. In other words, the previous
        upper boundary value is abandoned.

        However when merely applying given boundaries, the output intervals should be
        exactly where the values belong according to the given boundaries and does not
        have to cover the entire value space. Users may only pass in a few values
        to transform into intervals, forcing the largest interval to have inf may generate
        intervals that did not exist.

    Returns
    -------
    intervals_str: numpy.array, shape (number of examples,)
        Discretized result. The array of intervals that represented by strings.
    """
    intervals = _assign_interval_base(x, boundaries, force_inf=force_inf)
    # use join rather than use a+delimiter+b makes this line faster
    intervals_str = np.array(
        [delimiter.join((str(a), str(b))) for a, b in zip(intervals[:, 0],
                                                          intervals[:, 1])]
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
    n_j = column_unique.shape[0]  # number of columns, number of indexes (rows)

    # Sample frequencies of each combination of index vector and column vector
    # Consider the value array has two hierarchical indexes (index & column)
    dot_multiply = np.stack((index, column), axis=1)
    dot_multiply, value = np.unique(dot_multiply, axis=0, return_counts=True)

    # generate pivot table, there may be duplicated rows for index vector
    zeros = np.zeros(dot_multiply.shape[0])
    pivot_table = np.vstack(
        [np.where(dot_multiply[:, 1] == j, value, zeros) for j in column_unique]
    ).T

    # merge the pivot table rows with the same index value
    lowers = dot_multiply[:, 0]
    pivot_table = np.vstack(
        [pivot_table[lowers == i, :].sum(axis=0) for i in index_unique]
    )
    return pivot_table, column_unique, index_unique


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
    boundaries = np.array(
        list(set(delimiter.join(np.unique(vector)).split(delimiter))))
    boundaries = boundaries[(boundaries != '-inf') &
                            (boundaries != 'inf')].astype(float)
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
