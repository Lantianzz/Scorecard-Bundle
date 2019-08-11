# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9  12:03:43 2018
Updated on Thu Dec  13 15:35:00 2018

@author: zhanglt

Python module for standard scorecard modeling 
"""

from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

def _applyScoreCard(scorecard, factor_name, factor_column):
    """Apply the scorecard to a column. Return the score for each value.
    
    Parameters
    ----------
    scorecard: pandas.DataFrame, the scorecard.   
    
    factor_name: string, the name of the feature.
    
    factor_column: pandas.Series, the values of the feature.
    """
    score_rules = scorecard[scorecard['factor']==factor_name]
    interval_index = pd.IntervalIndex(score_rules['value'].values)
    return pd.Series([score_rules.iloc[interval_index.get_loc(factor_value),:]['score'] for factor_value in factor_column.values])

def _str2interval(str_interval):
    """convert string interval (e.g. (-0.1, 1.0]) to pandas.Interval
    This is useful when the scorecard is load from local file and all intervals become string.
    """
    return pd.Interval(
                float(str_interval.split()[0].replace('(','').replace(')','').replace('[','').replace(']','').replace(',','').replace(' ','')), 
                float(str_interval.split()[1].replace('(','').replace(')','').replace('[','').replace(']','').replace(',','').replace(' ','')),
             closed='right')

class LogisticRegressionScoreCard(BaseEstimator, TransformerMixin):
    """Take woe-ed features, fit a regression and turn it into a scorecard
    pandas0.23.4 should be installed.
    
    Parameters
    ----------
    woe_transformer: WOE transformer object from WOE module.

    C:  float, optional(Default=1.0)
        regularization parameter in linear regression. Default value is 1. 
        A smaller value implies more regularization.
        See details in scikit-learn document.

    class_weight: dict, optional(default=None)
    {class_label: weight}
    weights for each class of samples in linear regression. 
        This is to deal with imbalanced training data. 
        Default is 'auto'. This will aotumatically use class_weight from 
        scikit-learn to calculate the weights. The equivalent codes are:
        >>> from sklearn.utils import class_weight
        >>> class_weights2 = class_weight.compute_class_weight('balanced', 
                                                              np.unique(y), y)


    random_state: random seed in linear regression. Default is None.
        See details in scikit-learn document.

    PDO: Points to double odds. One of the parameters of Scorecard.
        Default value is 20. A positive value means the higher the 
        scores, the lower the probability of y being 1. A negative 
        value means the higher the scores, the higher the probability
        of y being 1.

    basePoints: the score for base odds(# of y=1/ # of y=0). Default is 100.

    decimal: Control the number of decimals that the output scores have.
        Default is 0 (no decimal)

    start_points: There are two types of scorecards, with and without start points.
        True means the scorecard will have a start poitns. Default is False.

    output_option: Controls the output format of scorecard. For now 'excel' is 
         the only option.

    output_path: The location to save the scorecard. e.g. r'D:\\Work\\jupyter\\'
         Default is None.          

    Attributes
    ---------- 
    woe_df_: pandas.DataFrame, the scorecard.
    
    AB_ : A and B when converting regression to scorecard.
    
    Code Examples
    ----------    
    import sys
    sys.path.append(r'D:\BaiduNetdiskDownload\0.codes\Algorithm') #put codes in this location
    import ChiMerge as cm
    import ScoreCard as sc
    import WOE
    import ModelEvaluation as me
    import Score2Rating as rating
    
    # chi2
    trans_chi2 = cm.ChiMerge(max_intervals=6,initial_intervals=100)
    result1 = trans_chi2.fit_transform(df, y_train_binary)
    #trans_chi2.A_dict
    
    # woe
    trans_woe = WOE.WOE(max_features=20)
    result2 = trans_woe.fit_transform(result1, y_train_binary)
    
    # scorecard 
    scorecard = sc.ScoreCard(woe_transformer=trans_woe, decimal=0, PDO=-20, basePoints=100)
    scorecard.fit(result2, y_train_binary)
    scored_result = scorecard.predict(df)
    
    # evaluate
    output_path = r'D:\\Work\\jupyter\\'
    me.ks_stat(y_train_binary.values, scored_result['TotalScore'].values) 
    me.ks(y_train_binary.values, scored_result['TotalScore'].values, output_path=output_path) 
    me.roc(y_train_binary.values, scored_result['TotalScore'].values, output_path=output_path) 
    me.precision_recall(y_train_binary.values, scored_result['TotalScore'].values, output_path=output_path) 
    me.lift_curve(y_train_binary.values, scored_result['TotalScore'].values, output_path=output_path)
    
    me.plot_all(y_train_binary.values, scored_result['TotalScore'].values, output_path=output_path)
    
    # Convert scores to investor suitability rating (irrelevant to standard scorecard modeling)
    rating_result = rating.score2rating(scored_result['TotalScore'], y_train, output_path)  
    """     
    
    def __init__(self, woe_transformer, C=1, class_weight=None, random_state=None,
                 PDO=-10, basePoints = 60, decimal=0, start_points = False,
                 output_option='excel', output_path=None):
        
        self.woe_transformer = woe_transformer
        self.C = C
        self.class_weight = class_weight
        self.random_state = random_state
        self.PDO = PDO
        self.basePoints = basePoints
        self.output_option = output_option
        self.decimal = decimal
        self.output_path = output_path
        self.start_points = start_points
    
    def fit(self, X, y):

        positive, total = y.sum(), len(y)    
        self.baseOdds_ = positive / (total - positive) 
        self.p_ = self.baseOdds_  / (1 + self.baseOdds_)

        B = self.PDO/np.log(2)
        A = self.basePoints + B * np.log(self.baseOdds_)
        self.AB_ = (A, B)
        
        # concat vertically all woe values
        woe_list = [pd.concat([
                pd.Series(len(self.woe_transformer.result_dict_[col][0])*[col]),
                pd.Series(list(self.woe_transformer.result_dict_[col][0].keys())),
                pd.Series(list(self.woe_transformer.result_dict_[col][0].values()))
                ],axis=1).set_index(0) for col in X.columns]
        self.woe_df_ = pd.concat(woe_list,axis=0).rename(columns={1:'value',2:'woe'})

        # fit a regression
        lr = LogisticRegression(C=self.C, class_weight=self.class_weight,
                                random_state=self.random_state)
        lr.fit(X, y)
        
        # Calculate scores for each value in each feature, and the start scores
        beta_map = dict(zip(list(X.columns),lr.coef_[0,:]))
        self.woe_df_['beta'] = pd.Series(self.woe_df_.index).map(beta_map).values
        self.woe_df_ = self.woe_df_.reset_index().rename(columns={0:'factor'})
        self.startPoints_ = A - B * lr.intercept_[0]    
        
        if self.start_points is True:
            self.woe_df_['score'] = np.around(-B * self.woe_df_['beta'].values * self.woe_df_['woe'].values, 
                       decimals=self.decimal)
            startPoints = pd.DataFrame({'factor': ['StartPoints'],
                'value': [np.nan],
                'woe': [np.nan],
                'beta': [np.nan],
                'score': np.around(self.startPoints_, decimals=self.decimal)
                })
            #the scorecard
            self.woe_df_ = pd.concat([startPoints, self.woe_df_],axis=0,ignore_index=True)  
        elif self.start_points is False:  
            self.woe_df_['score'] = np.around(-B * self.woe_df_['beta'].values * self.woe_df_['woe'].values + self.startPoints_ / X.shape[1], 
                       decimals=self.decimal)
        
        # change the first and last boundaries into -inf and inf
        for factor in pd.unique(self.woe_df_.factor):
            id_min = self.woe_df_[self.woe_df_.factor == factor].index.min()
            id_max = self.woe_df_[self.woe_df_.factor == factor].index.max()
            
            self.woe_df_.iloc[id_min,1] = pd.Interval(-float('inf'),
                        self.woe_df_.iloc[id_min,1].right,
                        closed='right')
            
            self.woe_df_.iloc[id_max,1] = pd.Interval(
                    self.woe_df_.iloc[id_max,1].left,
                    float('inf'),
                        closed='right')

        output = self.woe_df_.copy()
        output['value'] = output.value.astype(str)
        # Output the scorecard
        if self.output_option == 'excel' and self.output_path is None:
            output.to_excel('scorecards.xlsx', index=False)
        elif self.output_option == 'excel' and self.output_path is not None:
            output.to_excel(self.output_path+'scorecards.xlsx', index=False)
        self._lr = lr
    def predict(self, X_beforeWOE, load_scorecard=None):
        """Apply the scorecard.
        
        Parameters
        ----------
        X_beforeWOE: pandas.DataFrame. Features of samples (before woe-transformation)   
        
        load_scorecard: pandas.DataFrame. If we want to use a modified scorecard
            rather than the one automatically generated, we can pass the scorecard
            we want to use using this parameter. Default is None.
        """        
        
        if load_scorecard is None:
            scorecard = self.woe_df_
        else:
            scorecard = load_scorecard               

        scored_result = pd.concat([_applyScoreCard(scorecard, col, X_beforeWOE[col]) for col in scorecard['factor'].drop_duplicates().values], 
                                   axis=1)
        scored_result.columns = scorecard['factor'].drop_duplicates().values
        scored_result['TotalScore'] = scored_result.sum(axis=1)
        return scored_result
        
    def predict_proba(self, X_beforeWOE, load_scorecard=None):
        """Apply the scorecard.
        
        Parameters
        ----------
        X_beforeWOE: pandas.DataFrame. Features of samples (before woe-transformation)   
        
        load_scorecard: pandas.DataFrame. If we want to use a modified scorecard
            rather than the one automatically generated, we can pass the scorecard
            we want to use using this parameter. Default is None.
        """        
        
        if load_scorecard is None:
            scorecard_df = self.woe_df_
        else:
            scorecard_df = load_scorecard               

        if type(X_beforeWOE) is not pd.DataFrame:
            X_beforeWOE = pd.DataFrame(X_beforeWOE, columns=list(scorecard_df['factor'].drop_duplicates().values))
        
        scored_result = pd.concat([_applyScoreCard(scorecard_df, col, X_beforeWOE[col]) for col in scorecard_df['factor'].drop_duplicates().values], 
                                   axis=1)
        scored_result.columns = scorecard_df['factor'].drop_duplicates().values
        scored_result['TotalScore'] = scored_result.sum(axis=1)
        return scored_result['TotalScore'].values            
        
        
        
        
        







        