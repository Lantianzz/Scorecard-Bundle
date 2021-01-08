from scorecardbundle.feature_discretization import ChiMerge as cm
from scorecardbundle.feature_discretization import FeatureIntervalAdjustment as fia
from scorecardbundle.feature_encoding import WOE as woe
from scorecardbundle.feature_selection import FeatureSelection as fs
from scorecardbundle.model_training import LogisticRegressionScoreCard as lrsc
from scorecardbundle.model_evaluation import ModelEvaluation as me
from datetime import datetime


from importlib import reload
import ScorecardExplainer as scexp
reload(scexp)

import pandas as pd
import numpy as np


X = pd.DataFrame(np.random.rand(1000,20),columns=[f'feature{i}' for i in range(1,21)])
y = pd.Series(np.random.choice([1,0,0,0],1000))

trans_cm = cm.ChiMerge(max_intervals=5, min_intervals=2, output_dataframe=True)
result_cm = trans_cm.fit_transform(X, y) 

trans_woe = woe.WOE_Encoder(output_dataframe=True)
result_woe = trans_woe.fit_transform(result_cm, y)

model = lrsc.LogisticRegressionScoreCard(trans_woe, C=0.1,PDO=-10, basePoints=1000, verbose=True)
model.fit(result_woe, y)
rules = model.woe_df_

result1 = model.predict(pd.DataFrame(np.random.rand(1000,20),columns=[f'feature{i}' for i in range(1,21)]))
result2 = model.predict(pd.DataFrame(np.random.rand(1000,20),columns=[f'feature{i}' for i in range(1,21)]))
result3 = model.predict(pd.DataFrame(np.random.rand(1000,20),columns=[f'feature{i}' for i in range(1,21)]))
result1.columns
result1['end_dt'] = datetime(2021,1,1)
result2['end_dt'] = datetime(2021,2,1)
result3['end_dt'] = datetime(2021,3,1)

feature_names = [f'feature{i}' for i in range(1,21)]
bins = np.array([10,100,500,700,1000])

result1['explainer_important_features'] = scexp.important_features(result1,feature_names,threshold_method=0.8)
result1['explainer_important_features'] = scexp.important_features(result1,feature_names,threshold_method=3)
result1['explainer_important_features'] = scexp.important_features(result1,feature_names,threshold_method='bins',bins=bins)

 
a = [1,2,3]
a[:3]

tem.iloc[:5,:].to_dict('records')



thresholds = result1['TotalScore'].values*0.8
[(e.shape,t) for e,t in zip(result1.values,thresholds)]

result1['TotalScore'].values

boundaries_diff_boolean = result1['TotalScore'].values.reshape(1,-1).T >= bins.reshape(1,-1) 
uppers = np.array([bins[b].max() for b in boundaries_diff_boolean])
tem = pd.DataFrame({
    'bins':uppers
    ,'score':result1['TotalScore'].values
    })

uppers = np.array([boundaries[b].min() for b in ~boundaries_diff_boolean])
    
a = 0.8
isinstance(0.8,float)
