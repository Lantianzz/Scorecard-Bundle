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

Scorecard-Bundle is a **high-level Scorecard modeling API** that is easy-to-use and **Scikit-Learn consistent**.  It covers the major steps of training a Scorecard model including feature discretization with ChiMerge, WOE encoding, feature evaluation with information value and collinearity, Logistic-Regression-based Scorecard model, and model evaluation for binary classification tasks. All the transformers and model classes in Scorecard-Bundle comply with Scikit-Learn‘s fit-transform-predict convention.

A complete example showing how to build a scorecard with Scorecard-Bundle: [Example Notebooks](https://scorecard-bundle.bubu.blue/Notebooks/)

See detailed and more reader-friendly documentation in **https://scorecard-bundle.bubu.blue/**

In Scorecard-Bundle, core algorithms in WOE/IV calculation and scorecard transformation were based on the methods introduced in Mamdouh Refaat's book '"Credit Risk Scorecards: Development and Implementation Using SAS"；ChiMerge was written based on Randy Kerber's paper "ChiMerge: Discretization of Numeric Attributes".

I developed Scorecard-Bundle in my private time, but its codes wouldn't be so good if my superior [Andyshi](https://github.com/andysda) hasn't been allowing me to use it in projects at work, if my colleages (e.g. [zeyunH](https://github.com/zeyunH)) hasn't been active in using it, or if users didn't report issues when they found bugs.  Thanks to everyone who helps to make Scorecard-Bundle better.

## Installation

**Installing the latest version [![Downloads](https://img.shields.io/pypi/v/scorecardbundle?color=orange)](https://img.shields.io/pypi/v/scorecardbundle?color=orange)  is strongly recommended** as every version either corrected known bugs or added useful functionality.  In principle, critical bugs are fixed as soon as they are revealed. Therefore please file an issue if you suspect the presence of a bug when using Scorecard-Bundle.

Note that Scorecard-Bundle depends on NumPy, Pandas, matplotlib, Scikit-Learn, and SciPy, which can be installed individually or together through [Anaconda](https://www.anaconda.com/)

- Pip: Scorecard-Bundle can be installed with pip:  `pip install --upgrade scorecardbundle` 

  ！**Note that the latest version may be not available at some pip mirror site** (e.g. *https://mirrors.aliyun.com/pypi/simple/*). Therefore in order to update to the latest version,  use the following command to specify the source as *https://pypi.org/project*

  ~~~bash
  pip install -i https://pypi.org/project --upgrade scorecardbundle
  ~~~

  

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

- [Fix] 2 rare but critical bugs has been fixed from V1.2.1. Therefore I strongly advised anyone who uses Scorecard-bundle to update their older versions. See details of the bugs in the updates log for V1.2.1.  Thanks to @ zeyunH for bringing one of the bugs to me.
- [Notice] In several functions of WOE and ChiMerge module,  vector outer product is used to get the boolean mask matrix between two vectors. This may cause memory error if the feature has too many unique values (e.g.  a feature whose sample size is 350,000 and number of unique values is 10,000  caused this error in a 8G RAM laptop when calculating WOE). The tricky thing is the error message may not be "memory error" and this makes it harder for user to debug ( the current error message could be `TypeError: 'bool' object is not iterable` or  `DeprecationWarning:  elementwise comparison failed`). 
- [Fix] When using V1.0.2, songshijun007 brought up an issue about the occuring of KeyError due to too few unique values on training set and more extreme values in the test set. This issue has been fixed from V1.1.0.  (issue url: https://github.com/Lantianzz/Scorecard-Bundle/issues/1#issue-565173725).

### Updates Log

#### V1.2.1

This is an emergency update to fix 2 related bugs that may be triggered in rare cases but are hard to debug for someone who is not familiar with the codes. Thanks to @ zeyunH for bring one of the bugs to me.

- feature_discretization:
  - [Fix] Add parameter `force_inf` to `scorecardbundle/utils/func_numpy.py/_assign_interval_base()` and related codes. This parameter controls Whether to force the largest interval's right boundary to be positive infinity. Default is True.
    - Bug description:
      - In the case when the largest boundary value `b_max`passed is larger than or equal to the maximum feature value, the largest interval output is originally (xxx, b_max]. In tasks like fitting ChiMerge where the output intervals are supposed to cover the entire value space (-inf ~ inf), this parameter `force_inf` should be set to True so that the largest interval will be overwritten from (xxx, b_max] to (xxx, inf]. In other words, the previous largest boundary value is abandoned.
      - In the old version of codes the adjustment stated above was applied in all tasks. However,  when merely applying the given boundaries, the output intervals should be exactly where the values belong according to the given boundaries and does not have to cover the entire value space. In this case forcing the largest interval to have inf may generate intervals that should not exist. For example, the passed boundary values are 0, 10, 20, 30, while the largest feature value is only 20. The old version would change the largest interval from (10, 20] to (10, inf], which should not exist given the boundaries.
    - Solution in V1.2.1: Set `force_inf=True` in tasks like fitting ChiMerge where we want the output intervals to cover the entire value space so that the largest interval will be fixed to cover infinity. Set `force_inf=False` in tasks like ChiMerge transform and Scorecard predict where we only need to transform feature values into intervals based on the given boundaries.
  - [Fix] When generating intervals with `_assign_interval_base` in ChiMerge `fit()`,  the largest interval will be overwritten from (xxx, b_max] to (xxx, inf] to cover the entire value range. However, previously the codes only perform this adjustment when the largest boundary value is equal to the maximum value of the data, while in practive the largest boundary may be larger due to rounding (e.g. the max value is 3.14159 and the threshold happend to choose this value and rounded up to 3.1316 due to the `decimal` parameter of ChiMerge). From V1.2.1, the condition has been changed to `>=` 
- model_training.LogisticRegressionScoreCard:
  - [Fix] Set `force_inf=False` in function `assign_interval_str` when calling Scorecard predict(). This is to avoid getting KeyError because the maximum interval adjustment mentioned above generates an interval that does not exist in the Scorecard rules.
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

- 展示如何训练评分卡模型的完整示例见[Example Notebooks](https://scorecard-bundle.bubu.blue/Notebooks/)

- 详细的、更友好的文档见**https://scorecard-bundle.bubu.blue/**

Scorecard-Bundle中WOE和IV的计算、评分卡转化等的核心计算逻辑源自《信用风险评分卡研究 —基于SAS的开发与实施》一书，该书籍由王松奇和林治乾翻译自Mamdouh Refaat的"Credit Risk Scorecards: Development and Implementation Using SAS"；而ChiMerge算法则是复现了原作者Randy Kerber的论文"ChiMerge: Discretization of Numeric Attributes"。

虽然我是用私人时间开发的Scorecard-Bundle，但如果不是我的上级 [Andyshi](https://github.com/andysda) 允许我在工作中使用它、如果不是我的同事 (e.g. [zeyunH](https://github.com/zeyunH)) 积极的使用和反馈、如果不是用户们在发现bug时候提出issue，Scorecard-Bundle的代码不会有现在这么好。感谢帮助Scorecard-Bundle变得更好的每一个人。

### 安装

**由于每次版本更新都在修复已知的bug或添加重要的新功能，强烈建议安装最新版本 [![Downloads](https://img.shields.io/pypi/v/scorecardbundle?color=orange)](https://img.shields.io/pypi/v/scorecardbundle?color=orange)** 。严重的bug原则上都会在被发现的第一时间修复，因此若在使用Scorecard-Bundle的过程中怀疑存在bug，欢迎在issue中记录。

注意，Scorecard-Bundle依赖NumPy, Pandas, matplotlib, Scikit-Learn, SciPy，可单独安装或直接使用[Anaconda](https://www.anaconda.com/)安装。

- Pip: Scorecard-Bundle可使用pip安装:  `pip install --upgrade scorecardbundle` 

  注意！**最新版本可能尚未被纳入一些镜像源网站** (e.g. *https://mirrors.aliyun.com/pypi/simple/*)。因此为了更新到最新版本，可以使用下面的命令，指定 *https://pypi.org/project*作为源

  ~~~bash
  pip install -i https://pypi.org/project --upgrade scorecardbundle
  ~~~

  

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
  - [Fix]添加参数 `force_inf` 到函数 `scorecardbundle/utils/func_numpy.py/_assign_interval_base()`及相关代码，此参数控制是否会强制最大的区间的右侧边界为正无穷，默认为True
    - Bug描述：
      - 当传入的最大阈值`b_max`大于等于特征数据的最大值时，输出的最大的区间原本是(xxx, b_max]，而fit ChiMerge计算离散化的阈值时，需要输出的区间覆盖整个值域(-inf ~ inf)，此时这个参数应该被设为True，使得最大区间被从 (xxx, b_max] 改为(xxx, inf]，相当于原有的最大阈值被弃用了。
      - 旧版本的代码在所有情况下都无差别的应用了上面的修改规则，然而，当仅仅在应用已知的阈值将数值型数据转化为分箱时，输出的区间应该只有数值所处的位置决定，此时若对最大区间进行调整，可能会导致出现于原阈值不符的区间。例如传入的阈值是0,10,20,30，传入的数据最大值仅有20，旧代码会将最大的区间由原本的(10, 20]修改为(10, inf]，而根据给定的阈值不应该存在(10, inf]这个区间；
    - 修复：添加此参数作为开关后，在fit ChiMerge这样希望输出的区间覆盖整个值域的任务中使用`force_inf=True`，这样可以按需修正最大区间使其覆盖到正无穷；在用ChiMerge做transform操作、或使用评分卡的predict()这样希望严格按照阈值输出区间的任务中，使用`force_inf=False`；
  - [Fix] 当在 ChiMerge `fit()`中，旧版代码只会在最大阈值等于数据最大值时作上面提到的调整，然而实践中可能出现四舍五入导致最大阈值大于最大值的情况 (e.g. 最大值为3.14159 ，而最大阈值正好选中了这个值且由于ChiMerge的`decimal`参数四舍五入到了3.1316)。因此从V1.2.1开始，生效的条件被改为了`>=` 
- 模型训练 model_training.LogisticRegressionScoreCard:
  - [Fix] predict()中为函数`assign_interval_str` 设置`force_inf=False`，避免原代码在最大阈值等于数据最大值时会擅自修改输出的最大区间，导致出现评分规则中不存在的区间，造成评分规则时的KeyError
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


