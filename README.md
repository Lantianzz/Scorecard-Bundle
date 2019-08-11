# Scorecard-Bundle

The one package you need for Scorecard modeling in Python | 评分卡建模尽在于此

- [English Document](#english-document)
- [中文文档  (Chinese Document)](#中文文档--chinese-document)

## English Document

**Scorecard-Bundle is a Python toolkit for Scorecard modeling of binary targets**. The transformer and model classes in Scorecard-Bundle **comply with Scikit-learn**‘s fit-transform-predict convention.

There is a three-stage plan for Scorecard-Bundle:

- Stage 1 (Have been covered in v1.0): Replicate all functions of convectional Scorecard modeling, including:
  - Feature discretization with Chi-Merge;
  - WOE transformation and IV calculation;
  - Scorecard modeling based on Logistic regression;
  - Model Evaluation (binary classification evaluation);
- Stage 2 (Will be covered in v2.0): Add additional functionality, including:
  - Feature selection criteria (predictability + co-linearity + explainability);
  - Model scores discretization (if ratings are required);
  - Model Rating Evaluation (clustering quality evaluation);
  - Add discretization methods other than ChiMerge;
  - Add support for Scorecard based on algorithms other than Logistic Regression;
- Stage 3 (Will be covered in v3.0): Automate the modeling process, including:
  - Automatically select proper discretization methods for different features;
  - Automatically perform hyper-parameter tuning for LR-based Scorecard;
  - Automatically perform feature selection with consideration of predictability, co-linearity and explainability;
  - Provide an model pipeline that takes the input features, perform all the tasks (discretization, woe, etc.) inside it and return the scored samples and Scorecard rules. This simplify the modeling process to one line of code `model.fit_predict(X, y)`;

<img src="https://github.com/Lantianzz/ScorecardBundle/blob/master/pics/framework.svg">

## Quick Start

### Installment

- Pip: Scorecard-Bundle can be installed with pip `pip install scorecardbundle` 

- Manually: Down codes from github `<https://github.com/Lantianzz/Scorecard-Bundle>` and import them directly:

  ~~~python
  import sys
  sys.path.append('E:\Github\Scorecard-Bundle') # add path that contains the codes
  from scorecardbundle.feature_discretization import ChiMerge as cm
  from scorecardbundle.feature_encoding import WOE as woe
  from scorecardbundle.model_training import LogisticRegressionScoreCard as lrsc
  from scorecardbundle.model_evaluation import ModelEvaluation as me
  ~~~

### Usage

An usage example can be found in 



## Documentation



## Update Log

### Updates in v0.5

- ChiMerge:
  - Rewrite everything with Numpy (basically no Pandas at all). Now no error would be raised during training, even with unbalanced samples where the old implementation usually crash.

- WOE
  - Rewrite everything with Numpy. The code efficiency is boosted due to matrix computation;
  - The feature selection function is removed from WOE and will be included in an independent feature selection submodule;

### Updates in v0.4

- ChiMerge：
  - When the distribution of a feature is heavily unbalanced (e.g. most values are the same), pandas.qcut will crash. Thus we will switch to pandas.cut durng the above circumstances.
- ModelEvaluation:
  - Fixed a bug in lift curve. WNow the codes can generalize better.

- Scorecard
  - Add predict_proba() function to return scores only
  - Modify predict() and predict_proba() so that they support numpy array as input.

### Updates in v0.3

- ChiMerge
  - Fix a bug in ChiMerge that caused errors when bining data with pandas.qcut. If there are too many decimals in the minimum value of the column (e.g. 10 decimals), this minimum value would become the left boundary of the smallest interval procuced by qcut. The problem is that all intervals produced by qcut  are open on the left and close on the right. This means the smallest interval will not contain this minimum value.  To fix this, just round the column with pands.Series.round() before applying qcut.
  - Add a parameter `min_intervals`. When we don't want any feature droppped due to lack of predictability, we can use this parameter to make it happen.
- Scorecard
  - When using scorecard the data ranges may exceed those encountered in training, thus now the lowest and highest boundaries for each feature is set to negative infinity and positive infinity respectively.
- ModelEvaluation
  - If this module is run in jupyter notebook, the charts it saved to local used to be blank. This bug is fixed.

### Updates in v0.2

- Fix errors in notes. E.g. the default criterion for corr should be 0.6 rather than 0.7;
- Add example of using sklearn.utils.class_weight;
- Make Sure most default parameters  are optimal for Suitability scorecard model; 



## 中文文档  (Chinese Document)