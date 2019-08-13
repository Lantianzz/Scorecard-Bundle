# Scorecard-Bundle

An High-level Scorecard Modeling API | 评分卡建模尽在于此

- [English Document](#english-document)
  - [Installment](#installment)
  - [Usage](#usage)
- [中文文档  (Chinese Document)](#中文文档--chinese-document)
  - [安装](#安装)
  - [使用](#使用)

- [API Guide](#api-guide)
  - [Feature discretization](#feature-discretization)
  - [Feature encoding](#feature-encoding)
  - [Feature selection](#feature-selection)
  - [Model training](#model-training)
  - [Model evaluation](#model-evaluation)

## English Document

Scorecard-Bundle is a **high-level Scorecard modeling API** that is easy-to-use and **Scikit-Learn consistent**. The transformer and model classes in Scorecard-Bundle comply with Scikit-Learn‘s fit-transform-predict convention.

There is a three-stage plan for Scorecard-Bundle:

- Stage 1 (Have been covered in v1.0): Replicate all functions of convectional Scorecard modeling, including:
  - Feature discretization with Chi-Merge;
  - WOE transformation and IV calculation;
  - Feature selection based on IV and Pearson Correlation Coefficient;
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

### Installment

- Pip: Scorecard-Bundle can be installed with pip:  `pip install scorecardbundle` 

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

- Like Scikit-Learn, Scorecard-Bundle basiclly have two types of obejects, transforms and predictors. They comply with the fit-transform and fit-predict convention;
- An usage example can be found in https://github.com/Lantianzz/Scorecard-Bundle/blob/master/examples/Example_Basic_scorecard_modeling_with_Scorecard-Bundle.ipynb
- See more details in API Guide;

## 中文文档  (Chinese Document)

Scorecard-Bundle是一个基于Python的高级评分卡建模API，实施方便且符合Scikit-Learn的调用习惯，包含的类均遵守Scikit-Learn的fit-transform-predict习惯。

Scorecard-Bundle有三个阶段的开发计划：

- 阶段一 （已在v1.0.0中完成）：实现传统评分卡建模所的主要功能，包括：
  - 基于卡方分箱（Chi-Merge）的特征离散化；
  - WOE编码和IV计算；
  - 基于IV和皮尔森相关系数的特征筛选；
  - 基于逻辑回归的评分卡模型训练；
  - 模型评估（二元分类问题）。

- 阶段二（将在v2.0.0中完成）：补充更多功能，包括：
  - 全面的特征筛选指标（预测力+共线性+可解释性）；
  - 模型评分的离散化（如果需要评级）；
  - 模型评级的评估（聚类质量评价指标）；
  - 增加除Chi-Merge外的其他特征离散化算法；
  - 增加评分卡对除逻辑回归外的其他算法的支持。

- 阶段3 （将在v3.0.0中完成）：建模过程自动化，包括：
  - 自动为不同特征选择合适的离散化算法；
  - 自动为基于逻辑回归的评分卡调优超参数；
  - 根据特征预测力、共线性和可解释性评价指标，自动实施特征筛选；
  - 将建模流程组装成pipeline，使离散化、编码等任务在内部运行，pipeline直接返回评分结果和评分规则。这将模型训练过程简化为一行代码`model.fit_predict(X, y)`。

<img src="https://github.com/Lantianzz/ScorecardBundle/blob/master/pics/framework.svg">

### 安装

- Pip: Scorecard-Bundle可使用pip安装:  `pip install scorecardbundle` 

- 手动: 从Github下载代码`<https://github.com/Lantianzz/Scorecard-Bundle>`， 直接导入:

  ```python
  import sys
  sys.path.append('E:\Github\Scorecard-Bundle') # add path that contains the codes
  from scorecardbundle.feature_discretization import ChiMerge as cm
  from scorecardbundle.feature_encoding import WOE as woe
  from scorecardbundle.model_training import LogisticRegressionScoreCard as lrsc
  from scorecardbundle.model_evaluation import ModelEvaluation as me
  ```

### 使用

- 与Scikit-Learn相似，Scorecard-Bundle有两中class，transformer和predictor，分别遵守fit-transform和fit-predict习惯；
- 使用示例参见 https://github.com/Lantianzz/Scorecard-Bundle/blob/master/examples/%E7%A4%BA%E4%BE%8B_%E4%BD%BF%E7%94%A8Scorecard-Bundle%E8%BF%9B%E8%A1%8C%E5%9F%BA%E6%9C%AC%E7%9A%84%E8%AF%84%E5%88%86%E5%8D%A1%E5%BB%BA%E6%A8%A1.ipynb
- 详细用法参见API Guide;

## API Guide

#### Feature discretization

##### class: scorecardbundle.feature_discretization.ChiMerge.ChiMerge

ChiMerge is a discretization algorithm introduced by Randy Kerber in "ChiMerge: Discretization of Numeric Attributes". It can transform a numerical features into categorical feature or reduce the number of intervals in a ordinal feature based on the feature's distribution and the target classes' relative frequencies in each interval. As a result, it keep statistically significantly different intervals and merge similar ones.

###### Parameters

```mar
m: integer, optional(default=2)
    The number of adjacent intervals to compare during chi-squared test.

confidence_level: float, optional(default=0.9)
    The confidence level to determine the threshold for intervals to 
    be considered as different during the chi-square test.

max_intervals: int, optional(default=None)
    Specify the maximum number of intervals the discretized array will have.
    Sometimes (like when training a scorecard model) fewer intervals are 
    prefered. If do not need this option just set it to None.

min_intervals: int, optional(default=None)
    Specify the mininum number of intervals the discretized array will have.
    If do not need this option just set it to None.

initial_intervals: int, optional(default=100)
    The original Chimerge algorithm starts by putting each unique value 
    in an interval and merging through a loop. This can be time-consumming 
    when sample size is large. 
    Set the initial_intervals option to values other than None (like 10 or 100) 
    will make the algorithm start at the number of intervals specified (the 
    initial intervals are generated using quantiles). This can greatly shorten 
    the run time. If do not need this option just set it to None.

delimiter: string, optional(default='~')
    The returned array will be an array of intervals. Each interval is 
    representated by string (i.e. '1~2'), which takes the form 
    lower+delimiter+upper. This parameter control the symbol that 
    connects the lower and upper boundaries.

output_boundary: boolean, optional(default=False)
    If output_boundary is set to True. This function will output the 
    unique upper  boundaries of discretized array. If it is set to False,
    This funciton will output the discretized array.
    For example, if it is set to True and the array is discretized into 
    3 groups (1,2),(2,3),(3,4), this funciton will output an array of 
    [1,3,4].
```

###### Attributes

```
boundaries_: dict
    A dictionary that maps feature name to its merged boundaries.
fit_sample_size_: int
    The sampel size of fitted data.
transform_sample_size_:  int
    The sampel size of transformed data.
num_of_x_:  int
    The number of features.
columns_:  iterable
    An array of list of feature names.
```

###### Methods

```
fit(X, y): 
    fit the ChiMerge algorithm to the feature.

transform(X): 
    transform the feature using the ChiMerge fitted.

fit_transform(X, y): 
    fit the ChiMerge algorithm to the feature and transform it.    
```

#### Feature encoding

##### class: scorecardbundle.feature_encoding.WOE.WOE_Encoder

Perform WOE transformation for features and calculate the information value (IV) of features with reference to the target variable y.

###### Parameters

```
epslon: float, optional(default=1e-10)
        Replace 0 with a very small number during division 
        or logrithm to avoid infinite value.       

output_dataframe: boolean, optional(default=False)
        if output_dataframe is set to True. The transform() function will
        return pandas.DataFrame. If it is set to False, the output will
        be numpy ndarray.
```

###### Attributes

```
iv_: a dictionary that contains feature names and their IV

result_dict_: a dictionary that contains feature names and 
    their WOE result tuple. Each WOE result tuple contains the
    woe value dictionary and the iv for the feature.
```

###### Methods

```
fit(X, y): 
        fit the WOE transformation to the feature.

transform(X): 
        transform the feature using the WOE fitted.

fit_transform(X, y): 
        fit the WOE transformation to the feature and transform it.         
```

#### Feature selection

##### function: scorecardbundle.feature_selection.FeatureSelection.selection_with_iv_corr()

Retrun a table of each feature' IV and their highly correlated features to help users select features.

##### Parameters

```
trans_woe: scorecardbundle.feature_encoding.WOE.WOE_Encoder object,
        The fitted WOE_Encoder object

encoded_X: numpy.ndarray or pandas.DataFrame,
        The encoded features data

threshold_corr: float, optional(default=0.6)
        The threshold of Pearson correlation coefficient. Exceeding
        This threshold means the features are highly correlated.
```

##### Return

```
result_selection: pandas.DataFrame,
        The table that contains 4 columns. column factor contains the 
        feature names, column IV contains the IV of features, 
        column woe_dict contains the WOE values of features and 
        column corr_with contains the feature that are highly correlated
        with this feature together with the correlation coefficients.
```

#### Model training

##### class: scorecardbundle.model_training.LogisticRegressionScoreCard

Take encoded features, fit a regression and turn it into a scorecard

###### Parameters

```
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
    is the symbol that connects the lower and upper boundaries.
```

###### Attributes

```
woe_df_: pandas.DataFrame, the scorecard.

AB_ : A and B when converting regression to scorecard.
```

###### Methods

```
fit(woed_X, y): 
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
```

#### Model evaluation

##### class: scorecardbundle.model_evaluation.ModelEvaluation.BinaryTargets

Model evaluation for binary classification problem.

###### Parameters

```
y_true: numpy.array, shape (number of examples,)
        The target column (or dependent variable).  

y_pred_proba: numpy.array, shape (number of examples,)
        The score or probability output by the model. The probability
        of y_true being 1 should increase as this value
        increases.    

output_path: string, optional(default=None)
        the location to save the plot, e.g. r'D:\\Work\\jupyter\\'.
```

###### Methods

```
ks_stat(): Return the k-s stat
plot_ks(): Draw k-s curve
plot_roc(): Draw ROC curve
plot_precision_recall(): Draw precision recall curve
plot_all(): Draw k-s, ROC curve, and precision recall curve
```



## 