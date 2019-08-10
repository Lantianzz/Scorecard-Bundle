# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 13:49:59 2018
Updated on Thu Dec  13 15:35:00 2018

@author: zhanglt

按照survey_rating中问卷评级的分布为为评分（score)分档, 返回每个样本的模型评级
"""

from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn-colorblind')
plt.rcParams['font.sans-serif'] = ['SimHei'] #解决中文显示问题，设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False #解决保存图像负号'-'显示为方块的问题
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 120

font_text = {'family':'SimHei',
        'weight':'normal',
         'size':12,
        } # font for notmal text

font_title = {'family':'SimHei',
        'weight':'bold',
         'size':16,
        } # font for title


def confutionMatrix(model_rating, survey_rating, output_path):

    model_map = dict(zip(
    	range(1,6,1),
    	['模型c1(%)','模型c2(%)','模型c3(%)','模型c4(%)','模型c5(%)']
    	))

    survey_map = dict(zip(
    	range(1,6,1),
    	['问卷c1(%)','问卷c2(%)','问卷c3(%)','问卷c4(%)','问卷c5(%)']
    	))

    pivot_data = pd.DataFrame({
            '问卷评级':survey_rating.map(survey_map),
            '模型评级':model_rating.map(model_map),
            '数量':[1]*len(model_rating)
            })
    pivot_table = pd.pivot_table(pivot_data, values='数量', index='问卷评级', 
    	                         columns='模型评级', aggfunc=np.sum, fill_value=0)
    pivot_table_pct = (pivot_table*100/pivot_table.sum().sum()).round(2)

    # 精确匹配、模糊匹配、问评虚高、问评虚低
    exact_match = np.diagonal(pivot_table_pct.values).sum()
    fuzzy_match = sum([pivot_table_pct.iloc[:2,0].sum(),
                       pivot_table_pct.iloc[0:3,1].sum(),
                       pivot_table_pct.iloc[1:4,2].sum(),
                       pivot_table_pct.iloc[2:5,3].sum(),
                       pivot_table_pct.iloc[3:5,4].sum()])
    survey_2higher = sum([pivot_table_pct.iloc[2:5,0].sum(),
                          pivot_table_pct.iloc[3:5,1].sum(),
                          pivot_table_pct.iloc[4:5,2].sum()])
    survey_2lower = sum([pivot_table_pct.iloc[0,2:5].sum(),
                         pivot_table_pct.iloc[1,3:5].sum(),
                         pivot_table_pct.iloc[2,4:5].sum()])
    
    # 将混淆矩阵和匹配情况写入excel
    wb = Workbook()
    sheet = wb.active
    sheet.title = '混淆矩阵'
    sheet['A1'] = '混淆矩阵：模型评级 vs 问卷评级 (占总数的百分比）'
    for r in dataframe_to_rows(pivot_table_pct, index=True, header=True):
        sheet.append(r)
    sheet['A10'], sheet['B10'] = '精确匹配(%)', exact_match
    sheet['A11'], sheet['B11'] = '模糊匹配(%)', fuzzy_match
    sheet['A12'], sheet['B12'] = '问评虚高(%)', survey_2higher
    sheet['A13'], sheet['B13'] = '问评虚低(%)', survey_2lower
    wb.save(output_path+'混淆矩阵.xlsx')

    print(pivot_table_pct,
    	'\n 精确匹配(%)', exact_match,
		'\n 模糊匹配(%)', fuzzy_match,
		'\n 问评虚高(%)', survey_2higher,
		'\n 问评虚低(%)', survey_2lower,
		'\n 模型评级分布：\n', model_rating.value_counts().sort_index() / len(model_rating))

def score2rating_norm(score, mean, std):
    """classify scores into ratings using adjusted 3 sigma principle"""
    if mean - std < score < mean:
        return 2
    elif mean<= score < mean + std:
        return 3
    elif mean + std <= score < mean + 2 * std:
        return 4
    elif score >= mean + 2 * std:
        return 5
    elif score <= mean - std:
        return 1

def score2rating_norm2(score, mean, std):
    """classify scores into ratings using adjusted 3 sigma principle"""
    if mean - std < score < mean:
        return 3
    elif mean<= score < mean + 2 *std:
        return 4
    elif score >= mean + 2 * std:
        return 5
    elif mean - 2 * std< score <= mean - std:
        return 2        
    elif score <= mean - 2 * std:
        return 1

def score2rating_userDefined(score, th1, th2, th3, th4):
    """classify scores into ratings using user-defined thresholds"""
    thresholds = [th1, th2, th3, th4]
    if thresholds[0] < score < thresholds[1]:
        return 2
    elif thresholds[1]<= score < thresholds[2]:
        return 3
    elif thresholds[2] <= score < thresholds[3]:
        return 4
    elif score >= thresholds[3]:
        return 5
    elif score <= thresholds[0]:
        return 1

def score2rating(score, survey_rating, output_path, method='distribution', thresholds=None, survey_score=None, bins=30):
    """按照survey_rating中问卷评级的分布为为评分（score)分档。
    返回每个样本的模型评级

    Parameters
    ----------
    score: pandas.Series, shape (number of examples,)
           模型输出的概率或评分.
    
    survey_rating: pandas.Series, shape (number of examples,)
           样本的问卷评级
    output_path: 保存混淆矩阵的路径。例如r'D:\\Work\\jupyter\\'

    method: How to convert scores into ratings. Default is 'distribution'.
    	'distribution': bin scores into ratings depend on the scores' own distribution. 
                        Use this when we want the rating distribution skewed to the right. 

    				Devote the mean and standard deviation of scores by u and std, 
    				the 3 sigma principle states that the probability of samples falling 
    				into (u-std, u+std) is 65.26%, the probability of samples falling 
    				into (u-2*std, u+2*std) is 95.44%. Since 3 sigma principle applys to 
    				normal distribution, while the scores output by our risk tolerance model
    				are right-skewed (fat tail is on the right, the top of the curve leans on left), 
    				we adjust 3 sigma principle to our problem.

    				Mean is more affected by extreme values, thus it is replaced by median. 
    				Considering the scores output by our risk tolerance model are right-skewed 
    				(fat tail on the right), we define 1 to 5 ratings as follows:
    				    if u - std < score < u:
   						    return 2
   						elif u<= score < u + std:
   						    return 3
   						elif u + std <= score < u + 2 * std:
   						    return 4
   						elif score >= u + 2 * std:
   						    return 5
   						elif score <= u - std:
   						    return 1

        'distribution2': Similar with distribution. 
                         Use this when we want the rating distribution skewed to the left. 

                    Devote the mean and standard deviation of scores by u and std, 
                    the 3 sigma principle states that the probability of samples falling 
                    into (u-std, u+std) is 65.26%, the probability of samples falling 
                    into (u-2*std, u+2*std) is 95.44%. Since 3 sigma principle applys to 
                    normal distribution, while the scores output by our risk tolerance model
                    are right-skewed (fat tail is on the right, the top of the curve leans on left), 
                    we adjust 3 sigma principle to our problem.

                    Mean is more affected by extreme values, thus it is replaced by median. 
                    Considering the scores output by our risk tolerance model are left-skewed 
                    (fat tail on the left), we define 1 to 5 ratings as follows:
                    if mean - std < score < mean:
                        return 3
                    elif mean<= score < mean + 2 *std:
                        return 4
                    elif score >= mean + 2 * std:
                        return 5
                    elif mean - 2 * std< score <= mean - std:
                        return 2        
                    elif score <= mean - 2 * std:
                        return 1

    	'survey rating': bin scores into ratings depend on survey rating's distribution.
    	'user defined': bin scores into ratings using user-defined threolds.

	threolds: if method is set to 'user defined', here we need a threold list with 4 values.
				Defaul value is None.

	survey_score: survey scores. If this parameter is not none, a histogram showing the score distributions
			of model and survey will be produced. Default is None.
    
    bins: number of bins in the histogram that shows the score distributions of model and survey.
    """

    # 为评分分档
    if method == 'survey rating':
    	# 按问卷评级的分布为评分分档
    	q_list = [0] + list((survey_rating.value_counts()/len(survey_rating)).sort_index().iloc[:-1].cumsum()) + [1]
    	intervals = pd.Series(pd.qcut(score.values, q_list, precision=5, duplicates='drop'))

    	interval_map = dict(zip(
    		pd.Series(pd.unique(intervals)).sort_values().astype(str),
    		['模型c1(%)','模型c2(%)','模型c3(%)','模型c4(%)','模型c5(%)']
    		))

    	print('\n maps',pd.DataFrame.from_dict(interval_map, orient='index'))
    	interval_map_number = dict(zip(
    		pd.Series(pd.unique(intervals)).sort_values().astype(str),
    		range(1,6,1)
    		))
    	model_rating = intervals.astype(str).map(interval_map)
    	model_rating_number = intervals.astype(str).map(interval_map_number)

    elif method == 'distribution':
    	# 按评分自身分布分档，右偏
    	mean, std = score.median(), score.std()
    	model_rating_number= score.apply(score2rating_norm, args=(mean, std))
    	model_map = dict(zip(
    		range(1,6,1),
    		['模型c1(%)','模型c2(%)','模型c3(%)','模型c4(%)','模型c5(%)']
    		))
    	model_rating = model_rating_number.map(model_map)
    	print('Thresholds: ',[mean - std, mean, mean + std, mean + 2 * std])

    elif method == 'distribution2':
        # 按评分自身分布分档，左偏
        mean, std = score.median(), score.std()
        model_rating_number= score.apply(score2rating_norm2, args=(mean, std))
        model_map = dict(zip(
            range(1,6,1),
            ['模型c1(%)','模型c2(%)','模型c3(%)','模型c4(%)','模型c5(%)']
            ))
        model_rating = model_rating_number.map(model_map)
        print('Thresholds: ',[mean- 2 * std, mean - std, mean, mean + 2 * std])

    elif method == 'user defined':
    	# 按自定义的阈值分档
    	mean, std = score.median(), score.std()
    	model_rating_number= score.apply(score2rating_userDefined, args=(thresholds))
    	model_map = dict(zip(
    		range(1,6,1),
    		['模型c1(%)','模型c2(%)','模型c3(%)','模型c4(%)','模型c5(%)']
    		))
    	model_rating = model_rating_number.map(model_map)
    	print('Thresholds: ',thresholds)

    # 混淆矩阵
    survey_map = dict(zip(
    	pd.Series(pd.unique(survey_rating)).sort_values(),
    	['问卷c1(%)','问卷c2(%)','问卷c3(%)','问卷c4(%)','问卷c5(%)']
    	))
    
    pivot_data = pd.DataFrame({
            '问卷评级':survey_rating.map(survey_map),
            '模型评级':model_rating.values,
            '数量':[1]*len(model_rating)
            })
    pivot_table = pd.pivot_table(pivot_data, values='数量', index='问卷评级', 
    	                         columns='模型评级', aggfunc=np.sum, fill_value=0)
    pivot_table_pct = (pivot_table*100/pivot_table.sum().sum()).round(2)
    print(pivot_table)
    
    # 精确匹配、模糊匹配、问评虚高、问评虚低
    exact_match = np.diagonal(pivot_table_pct.values).sum()
    fuzzy_match = sum([pivot_table_pct.iloc[:2,0].sum(),
                       pivot_table_pct.iloc[0:3,1].sum(),
                       pivot_table_pct.iloc[1:4,2].sum(),
                       pivot_table_pct.iloc[2:5,3].sum(),
                       pivot_table_pct.iloc[3:5,4].sum()])
    survey_2higher = sum([pivot_table_pct.iloc[2:5,0].sum(),
                          pivot_table_pct.iloc[3:5,1].sum(),
                          pivot_table_pct.iloc[4:5,2].sum()])
    survey_2lower = sum([pivot_table_pct.iloc[0,2:5].sum(),
                         pivot_table_pct.iloc[1,3:5].sum(),
                         pivot_table_pct.iloc[2,4:5].sum()])

    # 将混淆矩阵和匹配情况写入excel
    wb = Workbook()
    sheet = wb.active
    sheet.title = '混淆矩阵'
    sheet['A1'] = '混淆矩阵：模型评级 vs 问卷评级 (占总数的百分比）'
    for r in dataframe_to_rows(pivot_table_pct, index=True, header=True):
        sheet.append(r)
    sheet['A10'], sheet['B10'] = '精确匹配(%)', exact_match
    sheet['A11'], sheet['B11'] = '模糊匹配(%)', fuzzy_match
    sheet['A12'], sheet['B12'] = '问评虚高(%)', survey_2higher
    sheet['A13'], sheet['B13'] = '问评虚低(%)', survey_2lower

    wb.save(output_path+'混淆矩阵.xlsx')

    print(pivot_table_pct,
    	'\n 精确匹配(%)', exact_match,
		'\n 模糊匹配(%)', fuzzy_match,
		'\n 问评虚高(%)', survey_2higher,
		'\n 问评虚低(%)', survey_2lower,
		'\n 模型评级分布：\n', model_rating.value_counts().sort_index() / len(model_rating))

	# 评分分布：模型 vs 问卷
    if survey_score is not None:
        S_TotalScore = (score - score.min()) / (score.max() - score.min()) *100
        S_SURVEY_SCORE = (survey_score - survey_score.min()) / (survey_score.max() - survey_score.min()) *100
        S_TotalScore.hist(bins=bins,alpha=0.6)
        S_SURVEY_SCORE.hist(bins=bins,alpha=0.6)
        plt.legend(['模型评分','问卷评分'])
        plt.title('评分分布：模型 vs 问卷（均转化为0-100分以便比较）', fontdict=font_title)
        plt.savefig(output_path+r'评分分布.png',dpi=500,bbox_inches='tight')
        plt.show()

    # 返回每个样本的模型评级
    return model_rating_number







