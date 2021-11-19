# Scorecard-Bundle

[![Downloads](https://pepy.tech/badge/scorecardbundle)](https://pepy.tech/project/scorecardbundle)  [![Downloads](https://img.shields.io/pypi/v/scorecardbundle?color=orange)](https://img.shields.io/pypi/v/scorecardbundle?color=orange)

A High-level Scorecard Modeling API | 评分卡建模尽在于此

**Documentation page** | **文档页面**：**https://scorecard-bundle.bubu.blue/**



- [ReadMe](#readme)
  - [Introduction](#introduction)
  - [Installment](#installment)
  - [Important Notice](#important-notice)
  - [Updates Log](#updates-log)
- [读我](#读我)
  - [简介](#简介)
  - [安装](#安装)
  - [重要公告](#重要公告)
  - [更新日志](#更新日志)



## ReadMe

### Introduction

Scorecard-Bundle is a **high-level Scorecard modeling API** that is easy-to-use and **Scikit-Learn consistent**.  It covers the major steps to train a Scorecard model such as feature discretization with ChiMerge, WOE encoding, feature evaluation with information value and collinearity, Logistic-Regression-based Scorecard model, and model evaluation for binary classification tasks. All the transformer and model classes in Scorecard-Bundle comply with Scikit-Learn‘s fit-transform-predict convention.

A complete example showing how to build a scorecard with Scorecard-Bundle: [Example Notebooks](https://scorecard-bundle.bubu.blue/Notebooks/)

See detailed and more reader-friendly documentation in **https://scorecard-bundle.bubu.blue/**

In Scorecard-Bundle, core codes such as WOE/IV calculation and scorecard transformation were written based on Mamdouh Refaat's book '"Credit Risk Scorecards: Development and Implementation Using SAS"；ChiMerge was written based on Randy Kerber's paper "ChiMerge: Discretization of Numeric Attributes".

### Installment

Note that Scorecard-Bundle depends on NumPy, Pandas, matplotlib, Scikit-Learn, and SciPy, which can be installed individually or together through [Anaconda](https://www.anaconda.com/)

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
  from scorecardbundle.model_interpretation import ScorecardExplainer as mise
  ~~~

### Important Notice

- [Fix] 2 rare but critical bugs has been fixed from V1.2.1. Therefore I strongly advised anyone who uses Scorecard-bundle to update their older versions. See details of the bugs in the updates log for V1.2.1.  Thanks to @ zeyunH for bring one of the bugs to me.
- [Notice] In several functions of WOE and ChiMerge module,  vector outer product is used to get the boolean mask matrix between two vectors. This may cause memory error if the feature has too many unique values (e.g.  a feature whose sample size is 350,000 and number of unique values is 10,000  caused this error in a 8G RAM laptop when calculating WOE). The tricky thing is the error message may not be "memory error" and this makes it harder for user to debug ( the current error message could be `TypeError: 'bool' object is not iterable` or  `DeprecationWarning:  elementwise comparison failed`). 
- [Fix] When using V1.0.2, songshijun007 brought up an issue about the occuring of KeyError due to too few unique values on training set and more extreme values in the test set. This issue has been fixed from V1.1.0.  (issue url: https://github.com/Lantianzz/Scorecard-Bundle/issues/1#issue-565173725).

### Updates Log

#### V1.2.1

This is an emergency update to fix 2 related bugs that may be triggered in rare cases but are hard to debug for someone who is not familiar with the codes. Thanks to @ zeyunH for bring one of the bugs to me.

- feature_discretization:
  - [Fix] Add parameter `force_inf` to `scorecardbundle/utils/func_numpy.py/_assign_interval_base(), assign_interval_unique(), assign_interval_str()`
    - This parameter controls Whether to force the largest interval's right boundary to be positive infinity. Default is True.
    - In the case when an upper boundary is not smaller then the maximum value, the largest interval output will be (xxx, upper]. In tasks like fitting ChiMerge where the output intervals are supposed to cover the entire value space (-inf ~ inf), this parameter `force_inf` should be set to True so that the largest interval will be overwritten from (xxx, upper] to (xxx, inf]. In other words, the previous upper boundary value is abandoned.
    - However when merely applying given boundaries, the output intervals should be exactly where the values belong according to the given boundaries and does not have to cover the entire value space. Users may only pass in a few values to transform into intervals, forcing the largest interval to have inf may generate intervals that did not exist.
    - Therefore, set `force_inf=True` when fitting ChiMerge; Set `force_inf=False` when calling ChiMerge transform or Scorecard predict.
  - [Fix] When generating intervals with `_assign_interval_base` in ChiMerge `fit()`,  the largest interval will be overwritten from (xxx, upper] to (xxx, inf] to cover the entire value range. However, previously the codes only perform this when the upper boundary (one of the given thresholds) is equal to the maximum value of the data, while in practive the upper boundary may be larger due to rounding (e.g. the max value is 3.14159 and the threshold happend to choose this value and rounded up to 3.1316 due to the `decimal` parameter of ChiMerge). From V1.2.1, the condition has been changed to `>=` 
- model_training.LogisticRegressionScoreCard:
  - [Fix] Set `force_inf=False` in function `assign_interval_str` when calling Scorecard predict();
  - [Add] Add a sanity check against the Scorecard rules on the `X_beforeWOE` parameter of `LogisticRegressionScoreCard.predict()` . In the case when the Scorecard rules have features which are not in the passed features data, or the passed features data has features which are not in the Scorecard rules, an exception will be raised.

#### V1.2.0

- feature_discretization:
  - [Add] Add parameter `decimal` to class `ChiMerge.ChiMerge()`, which allows users to control the number of decimals of the feature interval boundaries.
  - [Add] Add data table to the feature visualization `FeatureIntervalAdjustment.plot_event_dist()`. 
  - [Add] Add function `FeatureIntervalAdjustment.feature_stat()` that computes the input feature's sample distribution, including the sample sizes, event sizes and event proportions of each feature value.

- feature_selection.FeatureSelection:
  - [Add] Add function `identify_colinear_features()` that identifies the highly-correlated features pair that may cause colinearity problem.
  - [Add] Add function `unstacked_corr_table()`  that returns the unstacked correlation table to help analyze the colinearity problem.

- model_training.LogisticRegressionScoreCard:
  - [Fix] Alter the `LogisticRegressionScoreCard` class so that it now accepts all parameters of `sklearn.linear_model.LogisticRegression` and its `fit()` fucntion accepts all parameters of the `fit()` of `sklearn.linear_model.LogisticRegression` (including `sample_weight`)
  - [Add] Add parameter `baseOdds` for `LogisticRegressionScoreCard`. This allows users to pass user-defined base odds (# of y=1 / # of y=0) to the Scorecard model. 
  
- model_evaluation.ModelEvaluation:
  - [Add] Add function `pref_table`, which evaluates the classification performance on differet levels of model scores . This function is useful for setting classification threshold based on precision and recall.

- model_interpretation:
  - [Add] Add  function`ScorecardExplainer.important_features()`to help interpret the result of a individual instance. This function indentifies features who contribute the most in pusing the total score of a particular instance above a threshold. 

#### V1.1.3

- [Fix] Fixed a few minor bugs and warnings detected by Spyder's Static Code Analysis.  V1.1.3 covers all major steps of creating a scorecard model. This version has been used in dozens of scorecard modeling tasks without being found any error/bug during my career as a data analyst.

#### V1.1.0

- [Fix] Fixed a bug in `scorecardbundle.feature_discretization.ChiMerge.ChiMerge` to ensure the output discretized feature values are continous intervals from negative infinity to infinity, covering all possible values. This was done by modifying  `_assign_interval_base` function and `chi_merge_vector` function;
- [Fix] Changed the default value of `min_intervals` parameter in `scorecardbundle.feature_discretization.ChiMerge.ChiMerge` from None to 1 so that in case of encountering features with only one unique value would not cause an error. Setting the default value to 1 is actually more consistent to the actual meaning, as there is at least one interval in a feature;
- [Add] Add `scorecardbundle.feature_discretization.FeatureIntervalAdjustment` class to cover the functionality related to manually adjusting features in feature engineering stage. Now this class only contains `plot_event_dist` function, which can visualize a feature's sample distribution and event rate distribution. This is to facilate feature adjustment decisions in order to obtain better explainability and predictabiltiy;

#### V1.0.2

- Fixed a bug in scorecardbundle.feature_discretization.ChiMerge.ChiMerge.transform(). In V1.0.1, The transform function did not run normally when the number of unique values in a feature is less then the parameter 'min_intervals'. This was due to an ill-considered if-else statement. This bug has been fixed in v1.0.2;



## 读我

### 简介

Scorecard-Bundle是一个基于Python的高级评分卡建模API，实施方便且符合Scikit-Learn的调用习惯，包含的类均遵守Scikit-Learn的fit-transform-predict习惯。Scorecard-Bundle包括基于ChiMerge的特征离散化、WOE编码、基于信息值（IV）和共线性的特征评估、基于逻辑回归的评分卡模型、以及针对二元分类任务的模型评估。

展示如何训练评分卡模型的完整示例见[Example Notebooks](https://scorecard-bundle.bubu.blue/Notebooks/)

详细的、更友好的文档见**https://scorecard-bundle.bubu.blue/**

Scorecard-Bundle中WOE和IV的计算、评分卡转化等的核心计算逻辑源自《信用风险评分卡研究 —基于SAS的开发与实施》一书，该书籍由王松奇和林治乾翻译自Mamdouh Refaat的"Credit Risk Scorecards: Development and Implementation Using SAS"；而ChiMerge算法则是复现了原作者Randy Kerber的论文"ChiMerge: Discretization of Numeric Attributes"。

### 安装

注意，Scorecard-Bundle依赖NumPy, Pandas, matplotlib, Scikit-Learn, SciPy，可单独安装或直接使用[Anaconda](https://www.anaconda.com/)安装。

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
  from scorecardbundle.model_interpretation import ScorecardExplainer as mise
  ```

### 重要公告

- [Fix] 从V1.2.1开始修复了两处罕见但重要的bug，因此强烈建议Scorecard-bundle的用户更新旧版本的代码。bug的细节请见V1.2.1的更新日志；感谢@ zeyunH 指出其中的一个bug；
- [Notice] WOE和ChiMerge模块的几处代码（例如WOE模块的woe_vector函数）中，利用向量外积获得两个向量间的boolean mask矩阵，当输入的特征具有较多的唯一值时，可能会导致计算此外积的时候内存溢出（e.g. 样本量35万、唯一值1万个的特征，已在8G内存的电脑上计算WOE会内存溢出），此时的报错信息未必是内存溢出，给用户debug造成困难（当前的报错信息可能是`TypeError: 'bool' object is not iterable`或`DeprecationWarning:  elementwise comparison failed`）；
- [Fix] 在使用V1.0.2版本时，songshijun007 在issue中提到当测试集存在比训练集更大的特征值时会造成KeyError。这处bug已被解决，自V1.1.0版本起已修复（issue链接https://github.com/Lantianzz/Scorecard-Bundle/issues/1#issue-565173725).

### 更新日志

#### V1.2.1

为了修复两处罕见的bug而紧急发布V1.2.1版本。下面的bug对于不熟悉代码的用户较难排查。感谢@ zeyunH 指出其中的一个bug

- 特征离散化feature_discretization:
  - [Fix]添加参数 `force_inf` 到函数 `scorecardbundle/utils/func_numpy.py/_assign_interval_base(), assign_interval_unique(), assign_interval_str()`
    - 此参数控制是否会强制最大的区间的右侧边界为正无穷，默认为True
    - 当最大区间的右侧边界不小于数据的最大值时，最大的区间原本是(xxx, upper]. 对于fit ChiMerge这样需要输出的区间覆盖整个值域(-inf ~ inf)的任务而言，这个参数应该被设为True，使得最大区间被从 (xxx, upper] 改为(xxx, inf]，即原有的右侧边界这个阈值被弃用了
    - 然而，当仅仅在应用已知的阈值时，输出的区间应该只有数值所处的位置决定，此时若对最大区间进行调整，可能会导致出现于原阈值不符的区间
    - 因此，在fit ChiMerge时使用`force_inf=True`，在用ChiMerge做transform操作、或使用评分卡的predict()时，使用`force_inf=False`
  - [Fix] 当在 ChiMerge `fit()`中使用`_assign_interval_base`生成区间时，会对最大区间进行调整，使其右侧边界变为正无穷。旧版代码只会在区间的右侧边界等于数据最大值时作调整，然而实践中可能出现四舍五入导致的右侧边界大于最大值的情况 (e.g. 最大值为3.14159 ，而右侧边界正好选中了这个值且由于ChiMerge的`decimal`参数四舍五入到了3.1316)。因此从V1.2.1开始，生效的条件被改为了`>=` 
- 模型训练 model_training.LogisticRegressionScoreCard:
  - [Fix] predict()中为函数`assign_interval_str` 设置`force_inf=False`
  - [Add] 添加了对传入的特征数据`X_beforeWOE` 的检查，当评分规则中存在特征数据没有的特征、或特征数据中存在评分规则没有的特征时，会抛出异常

#### V1.2.0

- 特征离散化 feature_discretization:
  - [Add] 为class `ChiMerge.ChiMerge()`添加参数 `decimal` , 允许用户控制输出的特征区间的边界的小数位数；
  - [Add] 为特征分布可视化添加分布数据表 `FeatureIntervalAdjustment.plot_event_dist()`；
  - [Add] 添加函数`FeatureIntervalAdjustment.feature_stat()`用于计算特征的分布，包括不同取值的样本分布、响应率分布等；

- 特征选择 feature_selection.FeatureSelection:
  - [Add] 添加函数 `identify_colinear_features()` 用于识别高度相关的特征，输出高度相关的特征中IV较低的特征清单；
  - [Add] 添加函数 `unstacked_corr_table()` ，输出特征相关性表用于分析共线性问题；

- 模型训练 model_training.LogisticRegressionScoreCard:
  - [Fix] 优化`LogisticRegressionScoreCard` class ，使其可接受`sklearn.linear_model.LogisticRegression`的任意参数、且其`fit()`函数可接受`sklearn.linear_model.LogisticRegression`的fit()函数的任意参数 (包括 `sample_weight`)
  - [Add] 为`LogisticRegressionScoreCard`添加参数`baseOdds` . 这允许用户传入自定义的base odds (# of y=1 / # of y=0)
  
- 模型评估 model_evaluation.ModelEvaluation:
  - [Add] 添加函数 `pref_table`, 用于评估不同水平的模型分数的分类表现（精确度、召回率、F1、样本比例等）。此函数可帮助用户基于分类表现选择分类阈值；

- 评分卡解释 model_interpretation:
  - [Add] 添加函数`ScorecardExplainer.important_features()`用于解释单个样本的模型结果。此函数可识别对模型结果较重要的特征

#### V1.1.3

- [Fix] 修复Spyder的Static Code Analysis功能检测出的几处小bug和warning。V1.1.3覆盖了评分卡建模的主要步骤，在我作为数据分析师的数十次评分卡建模中未发现错误或bug

#### V1.1.0 

- [Fix]修正scorecardbundle.feature_discretization.ChiMerge.ChiMerge，使得任意情况下输出的取值区间都是负无穷到正无穷的连续区间（通过修改_assign_interval_base和chi_merge_vector实现）；
- [Fix] 将scorecardbundle.feature_discretization.ChiMerge.ChiMerge中的min_intervals默认值由None改为1，更符合实际情况（实际至少能有一个区间），当遇到特征的唯一值仅有一个的极端情况时也能直接输出此类特征的原值；
- [Add] 增加scorecardbundle.feature_discretization.FeatureIntervalAdjustment类，覆盖了特征工程阶段手动调整特征相关的功能，目前实现了`plot_event_dist`函数，可实现样本分布和响应率分布的可视化，方便对特征进行调整，已获得更好的可解释性和预测力；


#### V1.0.2

- [Fix] 修复scorecardbundle.feature_discretization.ChiMerge.ChiMerge.transform()的一处bug。在V1.0.1中，当一个特征唯一值的数量小于'min_intervals'参数时，transform函数无法正常运行，这是一处考虑不周的if-else判断语句造成的. 此bug已经在v1.0.2中修复;


