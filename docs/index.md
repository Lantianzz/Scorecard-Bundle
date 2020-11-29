[![Downloads](https://pepy.tech/badge/scorecardbundle)](https://pepy.tech/project/scorecardbundle)  [![Downloads](https://img.shields.io/pypi/v/scorecardbundle?color=orange)](https://img.shields.io/pypi/v/scorecardbundle?color=orange)

An High-level Scorecard Modeling API | 评分卡建模尽在于此



## English Document

### Introduction

Scorecard-Bundle is a **high-level Scorecard modeling API** that is easy-to-use and **Scikit-Learn consistent**. The transformer and model classes in Scorecard-Bundle comply with Scikit-Learn‘s fit-transform-predict convention.

### Installment

- Pip: Scorecard-Bundle can be installed with pip:  `pip install --upgrade scorecardbundle` 

- Manually: Download codes from github `<https://github.com/Lantianzz/Scorecard-Bundle>` and import them directly:

  ~~~python
  import sys
  sys.path.append('E:\Github\Scorecard-Bundle') # add path that contains the codes
  from scorecardbundle.feature_discretization import ChiMerge as cm
  from scorecardbundle.feature_discretization import FeatureIntervalAdjustment as fia
  from scorecardbundle.feature_encoding import WOE as woe
  from scorecardbundle.feature_selection import FeatureSelection as fs
  from scorecardbundle.model_training import LogisticRegressionScoreCard as lrsc
  from scorecardbundle.model_evaluation import ModelEvaluation as me
  ~~~

### Usage

- Like Scikit-Learn, Scorecard-Bundle basiclly have two types of obejects, transforms and predictors. They comply with the fit-transform and fit-predict convention;
- An complete example showing how to build a scorecard with high explainability and good predictability with Scorecard-Bundle can be found in https://github.com/Lantianzz/Scorecard-Bundle/blob/master/examples/%5BExample%5D%20Build%20a%20scorecard%20with%20high%20explainability%20and%20good%20predictability%20with%20Scorecard-Bundle.ipynb
- See more details in API Guide;



## 中文文档  (Chinese Document)

### 简介

Scorecard-Bundle是一个基于Python的高级评分卡建模API，实施方便且符合Scikit-Learn的调用习惯，包含的类均遵守Scikit-Learn的fit-transform-predict习惯。

### 安装

- Pip: Scorecard-Bundle可使用pip安装:  `pip install --upgrade scorecardbundle` 

- 手动: 从Github下载代码`<https://github.com/Lantianzz/Scorecard-Bundle>`， 直接导入:

  ```python
  import sys
  sys.path.append('E:\Github\Scorecard-Bundle') # add path that contains the codes
  from scorecardbundle.feature_discretization import ChiMerge as cm
  from scorecardbundle.feature_discretization import FeatureIntervalAdjustment as fia
  from scorecardbundle.feature_encoding import WOE as woe
  from scorecardbundle.feature_selection import FeatureSelection as fs
  from scorecardbundle.model_training import LogisticRegressionScoreCard as lrsc
  from scorecardbundle.model_evaluation import ModelEvaluation as me
  ```

### 使用

- 与Scikit-Learn相似，Scorecard-Bundle有两种class，transformer和predictor，分别遵守fit-transform和fit-predict习惯；
- 完整示例展示如何训练可解释性和预测力俱佳的评分卡 https://github.com/Lantianzz/Scorecard-Bundle/blob/master/examples/%5BExample%5D%20Build%20a%20scorecard%20with%20high%20explainability%20and%20good%20predictability%20with%20Scorecard-Bundle.ipynb
- 详细用法参见API Guide;



