# -*- coding: utf-8 -*-
"""
Training the logistic regression based Scorecard model.

@author: Lantian ZHANG
"""

from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

from scorecardbundle.utils.func_numpy import assign_interval_str
from scorecardbundle.utils.func_numpy import interval_to_boundary_vector
from scorecardbundle.utils.func_numpy import map_np


# ============================================================
# Basic Functions
# ============================================================

def _apply_scorecard(scorecard, feature_name, feature_array, delimiter='~'):
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
    intervals = assign_interval_str(feature_array, boundaries, delimiter=delimiter, force_inf=False)
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

    baseOdds: float, optional(default=None)
        The ratio of the number of positive class(y=1) divided by that of negative class(y=0)
        Leave this parameter to None means baseOdds will be automatically calculated
        using the number of positive class divided by the number of negative class in y.

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
        is the symbol that connects the lower and upper boundaries..

    **kargs: other keyword arguments in sklearn.linear_model.LogisticRegression()
    
    Attributes
    ---------- 
    woe_df_: pandas.DataFrame, the Scorecard scoring rules.
            The table contains 5 columns (feature, value, woe, beta and score). 
            - 'feature' column: feature names
            - 'value' column: feature intervals (right-closed)
            - 'woe' column: WOE encodings of feature intervals
            - 'score' column: the score for the feature interval respectively

            An example would be as followed,
            feature value   woe         beta        score
            x1      30~inf  0.377563    0.631033    5.0
            x1      20~-30  1.351546    0.631033    37.0
            x1      -inf~20 1.629890    0.631033    -17.0
    
    AB_ : A and B when converting regression to scorecard.

    Methods
    -------
    fit(woed_X, y,**kargs): 
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
    
    def __init__(self, woe_transformer, C=1.0, class_weight=None,
                 PDO=-20, basePoints=100, baseOdds=None,
                 decimal=0, start_points = False,
                 output_option='excel', output_path=None, verbose=False,
                 delimiter='~', random_state=None, **kargs):
        
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
        self.verbose = verbose
        self.delimiter = delimiter
        self.baseOdds = baseOdds
        self.base_model_kargs = kargs
        self.fit_sample_size_ = None
        self.num_of_x_ = None
        self.columns_ = None
        self.p_ = None
        self.AB_ = None
        self.woe_df_ = None
        self.beta_map_ = None
        self.startPoints_ = None
        self.transform_sample_size_ = None

    def _init_fit(self, y):
        """Initialize the fitting of scorecard
        """
        # Basic settings of Scorecard
        positive, total = y.sum(), y.shape[0]
        if self.baseOdds is None:
            self.baseOdds = positive / (total - positive)
        self.p_ = self.baseOdds / (1 + self.baseOdds)
        B = self.PDO/np.log(2)
        A = self.basePoints + B * np.log(self.baseOdds)
        self.AB_ = (A, B)

        # Concat vertically all woe values
        # later this will become the scorecard rules table
        woe_list = [pd.concat([
           pd.Series([col]*len(self.woe_transformer.result_dict_[col][0])), # Name of x
           pd.Series(list(self.woe_transformer.result_dict_[col][0].keys())), # interval strings of x
           pd.Series(list(self.woe_transformer.result_dict_[col][0].values())) # woe values of x
           ],axis=1) for col in self.columns_]
        self.woe_df_ = pd.concat(woe_list, axis=0, ignore_index=True
                                 ).rename(columns={0: 'feature', 1: 'value', 2: 'woe'})

    def _get_w_b(self, features, y, **kargs):
        """Get the betas(W) and intercepts(b)"""
        # Initialize regression
        lr = LogisticRegression(C=self.C
                                , class_weight=self.class_weight
                                , random_state=self.random_state
                                , **self.base_model_kargs)

        # Fit a logistic regression
        lr.fit(features, y, **kargs)

        # Calculate scores for each value in each feature, and the start scores
        self.beta_map_ = dict(zip(list(self.columns_), lr.coef_[0, :]))
        self.intercept_ = lr.intercept_[0]

    def fit(self, woed_X, y, **kargs):
        """
        Parameters
        ----------
        woed_X: numpy.ndarray or pandas.DataFrame, shape (number of examples,
                                                     number of features)
            The woe encoded X.

        y: numpy.array or pandas.Series, shape (number of examples,)
            The target array (or dependent variable).

        **kargs: other keyword arguments in the fit()
            of sklearn.linear_model.LogisticRegression
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
                [''.join(('x', str(a))) for a in range(self.num_of_x_)]
                )  # column names (i.e. x0, x1, ...)
            features = woed_X
        else:
            raise TypeError('woed_X should be either numpy.ndarray or pandas.DataFrame')

        if isinstance(y, pd.Series):
            y = y.values
        elif isinstance(y, np.ndarray):
            pass
        else:
            raise TypeError('y should be either numpy.array or pandas.Series')

        # Initialize Scorecard fitting
        self._init_fit(y)

        # Get the betas(W) and intercepts(b)
        self._get_w_b(features, y, **kargs)

        # Rule table for Scoracard
        self.startPoints_ = self.AB_[0] - self.AB_[1] * self.intercept_
        self.woe_df_['beta'] = map_np(self.woe_df_.feature, self.beta_map_)
        if self.start_points is True:
            self.woe_df_['score'] = np.around(
                -self.AB_[1] * self.woe_df_['beta'].values * self.woe_df_['woe'].values,
                decimals=self.decimal)
            startPoints = pd.DataFrame({'feature': ['StartPoints'],
                'value': [np.nan],
                'woe': [np.nan],
                'beta': [np.nan],
                'score': np.around(self.startPoints_, decimals=self.decimal)
                })
            # Rule table
            self.woe_df_ = pd.concat([startPoints, self.woe_df_],
                                     axis=0,
                                     ignore_index=True)  
        elif self.start_points is False:
            self.woe_df_['score'] = np.around(
                -self.AB_[1] * self.woe_df_['beta'].values * self.woe_df_['woe'].values + self.startPoints_ / self.num_of_x_,
                decimals=self.decimal)

        # Output the scorecard
        if self.output_option == 'excel' and self.output_path is None:
            self.woe_df_.to_excel('scorecards.xlsx', index=False)
        elif self.output_option == 'excel' and self.output_path is not None:
            self.woe_df_.to_excel(self.output_path+'scorecards.xlsx', index=False)

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
            self.columns_ = X_beforeWOE.columns.values # column names
        elif isinstance(X_beforeWOE, np.ndarray):
            features = X_beforeWOE.T
            self.columns_ = np.array(
                [''.join(('x',str(a))) for a in range(X_beforeWOE.shape[1])]
                )  # column names (i.e. x0, x1, ...)
        else:
            raise TypeError('X_beforeWOE should be either numpy.ndarray or pandas.DataFrame')

        # Check whether the user choose to load a Scorecard rule table
        if load_scorecard is None:
            scorecard = self.woe_df_
        else:
            scorecard = load_scorecard

        # Exam the passed features data
        set_missing = set(scorecard['feature'].unique()) - set(self.columns_)
        set_extra = set(self.columns_) - set(scorecard['feature'].unique())
        if len(set_missing) > 0:
            raise Exception("Scorecard rules have features which are not in the passed features data:" + str(set_missing))
        elif len(set_extra) > 0:
            raise Exception("The passed features data has features which are not in the Scorecard rules:" + str(set_extra))
        else:
            pass  # Assertion passed

        # Apply the Scorecard rules
        scored_result = pd.concat(
            [pd.Series(_apply_scorecard(scorecard,
                                        name,
                                        x,
                                        delimiter=self.delimiter
                                        ),
                       name=name
                       ) for name, x in zip(self.columns_, features)],
            axis=1)
        scored_result['TotalScore'] = scored_result.sum(axis=1)
        if self.verbose:
            return scored_result
        else:
            return scored_result['TotalScore'].values
        