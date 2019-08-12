# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 2019
Updated on Mon Aug 12 2019

@authors: Lantian ZHANG <zhanglantian1992@163.com>

"""
import pandas as pd
import numpy as np

def selection_with_iv_corr(trans_woe, encoded_X, threshold_corr=0.6):
    """Calculate WOE and IV for 1 feature
    Parameters
    ----------
    trans_woe: scorecardbundle.feature_encoding.WOE.WOE_Encoder object,
            The fitted WOE_Encoder object

    encoded_X: numpy.ndarray or pandas.DataFrame,
            The encoded features data
   
    threshold_corr: float, optional(default=0.6)
            The threshold of Pearson correlation coefficient. Exceeding
            This threshold means the features are highly correlated.

    Return
    ----------
    result_selection: pandas.DataFrame,
            The table that contains 4 columns. column factor contains the 
            feature names, column IV contains the IV of features, 
            column woe_dict contains the WOE values of features and 
            column corr_with contains the feature that are highly correlated
            with this feature together with the correlation coefficients.
    """
    # if X is pandas.DataFrame, turn it into numpy.ndarray and 
    # associate each column array with column names.
    # if X is numpy.ndarray, 
    if isinstance(encoded_X, pd.DataFrame):
        data = encoded_X
    elif isinstance(X, np.ndarray):
        columns = np.array(
            [''.join(('x',str(a))) for a in range(self.num_of_x_)]
            ) #  # column names (i.e. x0, x1, ...)
        data = pd.DataFrame(encoded_X, columns=columns)
    else:
        raise TypeError('encoded_X should be either numpy.ndarray or pandas.DataFrame')
    
    corr_matrix = data.corr().reset_index().rename(
                                                columns={'index':'corr_with'})
    result_selection = pd.DataFrame.from_dict(
        trans_woe.iv_, orient='index'
        ).reset_index().rename(columns={'index':'factor',0:'IV'}
                                ).sort_values('IV', ascending=False)
    result_selection['woe_dict'] = [trans_woe.result_dict_.get(col)[0] for col in result_selection.factor]
    corr_mask = [((corr_matrix[corr_matrix.corr_with!=col][col].abs()>threshold_corr),
                  col) for col in result_selection.factor]
    result_selection['corr_with']  = [dict(
        zip(corr_matrix[corr_matrix.corr_with!=col].corr_with[mask].values, 
            corr_matrix[corr_matrix.corr_with!=col][col][mask].values)
        ) for mask,col in corr_mask]
    return result_selection