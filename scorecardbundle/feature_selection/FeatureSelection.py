# -*- coding: utf-8 -*-
"""Feature selection tools.

@authors: Lantian ZHANG
"""
import pandas as pd
import numpy as np

def selection_with_iv_corr(trans_woe, encoded_X, threshold_corr=0.6):
    """Retrun a table of each feature' IV and their highly correlated
    features to help users select features.
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
    elif isinstance(encoded_X, np.ndarray):
        columns = np.array(
            [''.join(('x',str(a))) for a in range(encoded_X.shape[1])]
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

def unstacked_corr_table(encoded_X,dict_iv):
    """Return the unstacked correlation table to help analyze the colinearity problem.

    Parameters
    ----------
    encoded_X: numpy.ndarray or pandas.DataFrame,
            The encoded features data

    dict_iv: python dictionary.
            The ditionary where the keys are feature names and values are the information values (iv)
   
    Return
    ----------
    corr_unstack: pandas.DataFrame,
            The unstacked correlation table

    """
    # if X is pandas.DataFrame, turn it into numpy.ndarray and 
    # associate each column array with column names.
    # if X is numpy.ndarray, 
    if isinstance(encoded_X, pd.DataFrame):
        data = encoded_X
    else:
        raise TypeError('encoded_X should be either pandas.DataFrame')

    corr_matrix = data.corr()
    corr_unstack = corr_matrix.unstack().reset_index()
    corr_unstack.columns = ['feature_a','feature_b','corr_coef']
    corr_unstack['abs_corr_coef'] = corr_unstack['corr_coef'].abs()
    corr_unstack = corr_unstack[corr_unstack['feature_a']!=corr_unstack['feature_b']].reset_index(drop=True)
    corr_unstack['iv_feature_a'] = corr_unstack['feature_a'].map(lambda x: dict_iv[x])
    corr_unstack['iv_feature_b'] = corr_unstack['feature_b'].map(lambda x: dict_iv[x])
    return corr_unstack.sort_values('abs_corr_coef',ascending=False)

def identify_colinear_features(encoded_X,dict_iv,threshold_corr=0.6):
    """Identify the highly-correlated features pair that may cause colinearity problem.

    Parameters
    ----------
    encoded_X: numpy.ndarray or pandas.DataFrame,
            The encoded features data

    dict_iv: python dictionary.
            The ditionary where the keys are feature names and values are the information values (iv)
   
    threshold_corr: float, optional(default=0.6)
            The threshold of Pearson correlation coefficient. Exceeding
            This threshold means the features are highly correlated.

    Return
    ----------
    features_to_drop_auto: python list,
            The features with lower IVs in highly correlated pairs.

    features_to_drop_manual: python list,
            The features with equal IVs in highly correlated pairs.

    corr_auto: pandas.DataFrame,
            The Pearson correlation coefficients and information values (IV) 
            of highly-correlated features pairs where the feature with lower IV
            will be dropped.

    corr_manual: pandas.DataFrame,
            The Pearson correlation coefficients and information values (IV) 
            of highly-correlated features pairs where the features have equal IV values
            and human intervention is required to choose the feature to drop.

    """
    # if X is pandas.DataFrame, turn it into numpy.ndarray and 
    # associate each column array with column names.
    # if X is numpy.ndarray, 
    if isinstance(encoded_X, pd.DataFrame):
        data = encoded_X
    else:
        raise TypeError('encoded_X should be either pandas.DataFrame')

    corr_matrix = data.corr()
    corr_unstack = corr_matrix.unstack().reset_index()
    corr_unstack.columns = ['feature_a','feature_b','corr_coef']
    corr_unstack = corr_unstack[corr_unstack['feature_a']!=corr_unstack['feature_b']].reset_index(drop=True)
    corr_unstack = corr_unstack[corr_unstack.corr_coef.abs()>threshold_corr].reset_index(drop=True)
    corr_unstack['iv_feature_a'] = corr_unstack['feature_a'].map(lambda x: dict_iv[x])
    corr_unstack['iv_feature_b'] = corr_unstack['feature_b'].map(lambda x: dict_iv[x])
    corr_unstack['to_drop'] = np.where(corr_unstack.iv_feature_a>corr_unstack.iv_feature_b,corr_unstack.feature_b,corr_unstack.feature_a)
    corr_manual = corr_unstack[corr_unstack['iv_feature_a']==corr_unstack['iv_feature_b']].reset_index(drop=True)
    corr_auto = corr_unstack[corr_unstack['iv_feature_a']!=corr_unstack['iv_feature_b']].reset_index(drop=True)
    features_to_drop_auto = list(corr_auto.to_drop.unique())
    features_to_drop_manual = list(corr_manual.to_drop.unique())
    return features_to_drop_auto, features_to_drop_manual, corr_auto, corr_manual