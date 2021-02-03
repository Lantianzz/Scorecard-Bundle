# -*- coding: utf-8 -*-
"""
Created on Thu Jan 8 2021
Updated on Thu Feb 3 2021

@authors: Lantian ZHANG <peter.lantian.zhang@outlook.com>

Identify the features who contribute the most in pusing the total score above a threshold.
"""
import pandas as pd
import numpy as np 

def _instance_important_features(scores,feature_names,threshold):
    """For one instance, indentify features who contribute the most 
    in pusing the instance's total score above a threshold.
    
    Parameters
    ----------
    scores: numpy.array, shape (number of features,)
        The feature values of one instance
    
    feature_names: python list
        The names of features 
    threshold: float or integer
    
    Returns
    -------
    ifeatures: python dictionary.
        A dictionary of importance features with feature names as keys and scores as values.
    """  
    tem = pd.DataFrame({
       'feature':feature_names
       ,'score':scores
       }).sort_values('score',ascending=False)
    tem['cum_score'] = tem['score'].cumsum()
    max_value = tem['cum_score'].max()
    threshold =  max_value if max_value<threshold else threshold # Make sure the threshold passed is no larger than the maximum value
    tem = tem[tem.score>0].reset_index(drop=True) # Exclude negative scores since they drag down the total score rather than push up
    tem = tem.iloc[0:tem.index[tem['cum_score']<threshold].max()+1,:] # Filter the features that contribute the most
    ifeatures = dict(zip(tem.iloc[:,0],tem.iloc[:,1])) # returned dictionary
    return ifeatures

def _instance_top_features(scores,feature_names,n):
    """For one instance, indentify the instance's n-highest-score features.
    
    Parameters
    ----------
    scores: numpy.array, shape (number of features,)
        The feature values of one instance
    
    feature_names: python list
        The names of features 
    
    n: integer
        integer in [1,number of features]. Top n important features will be returned
    
    Returns
    -------
    ifeatures: python dictionary.
        A dictionary of importance features with feature names as keys and scores as values.
    """  
    tem = pd.DataFrame({
       'feature':feature_names
       ,'score':scores
       }).sort_values('score',ascending=False)
    tem = tem.iloc[0:n,:] # 
    ifeatures = dict(zip(tem.iloc[:,0],tem.iloc[:,1]))
    return ifeatures

def important_features(scored_df,feature_names,col_totalscore='TotalScore',threshold_method=0.8, bins=None):
    """Indentify features who contribute the most in pusing the total score above a threshold.
    
    Parameters
    ----------
    scored_df: pandas.DataFrame, shape (number of instances,number of features)
        The dataframe that contains the both each feature's scores and total scores 
    feature_names: python list
        The names of features 
    col_totalscore: python string
        The name of the total score column. Default is 'TotalScore'
    threshold_method: float in (0,1) or string 'bins' or integer in [1,number of features].
        The method to get the thresholds to filter importance features. Default is 0.8
        
        - When threshold_method is a float in interval (0,1), the thresholds will be calculated as the 
        threshold_method percentage of total scores.
        
        - When threshold_method is a integer in [1,number of features]. Top n important features 
        will be returned.
        
        - When threshold_method=='bins', the thresholds will be determined by predefined bins 
        (defined in `bins` parameter).
        The largest bin value below the total score will be selected for each instance.
        The method 'bins' is not recommended 
        
    
    bins: numpy.array, shape (number of bins,).
        The predefined bins to bin the total score into intervals. This parameter is only used when
        threshold_method=='bins'. Default is None.
    Returns
    -------
    ifeatures_list: python list.
        A list of dictionaries of importance features with feature names as keys and scores as values.
    """
    
    # Filter features based on predefined bins
    if threshold_method=='bins': 
       mask_matrix = scored_df['TotalScore'].values.reshape(1,-1).T >= bins.reshape(1,-1) 
       thresholds = np.array([bins[b].max() for b in mask_matrix]) # Select the largest bin values below the total scores
    
    # Filter features based on percentage of total scores
    elif isinstance(threshold_method,float) and 0<threshold_method<1: 
       thresholds = scored_df[col_totalscore].values*threshold_method
    
    # Filter top n important features
    elif isinstance(threshold_method,int) and 1<=threshold_method<=len(feature_names): 
        ifeatures_list = [_instance_top_features(e,feature_names,threshold_method) for e in scored_df[feature_names].values]
        return ifeatures_list
    
	# Raise exception
    else:
        raise TypeError("Unsupported threshold_method. Valid values can be float in (0,1) or string 'bins' or integer in [1,number of features].")
    
    ifeatures_list = [_instance_important_features(e,feature_names,threshold) for e,threshold in zip(scored_df[feature_names].values,thresholds)]
    return ifeatures_list


