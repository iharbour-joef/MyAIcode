import glob
import os
import gc
import seaborn as sns
import os
import datetime
import sys
import joblib

import toad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import entropy
from sklearn.linear_model import LinearRegression

import category_encoders as ce
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.metrics import  f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

from sklearn.model_selection import  RandomizedSearchCV
from scipy import special, optimize

from gensim.models import word2vec
from scipy import spatial
from sklearn.cluster import KMeans
from sklearn import metrics
import math
import datetime
import time

#读取数据
train_userlabel=pd.read_csv('G:/data/User_Identification/训练数据/用户标签.csv')
train_userinformation=pd.read_csv('G:/data/User_Identification/训练数据/用户基础信息.csv')
train_usersubmission=pd.read_csv('G:/data/User_Identification/训练数据/用户投稿信息.csv')
train_userbehavior=pd.read_csv('G:/data/User_Identification/训练数据/用户行为信息.csv')

test_userlabel=pd.read_csv('G:/data/User_Identification/测试数据/用户id(无标签).csv')
test_userinformation=pd.read_csv('G:/data/User_Identification/测试数据/用户基础信息.csv')
test_usersubmission=pd.read_csv('G:/data/User_Identification/测试数据/用户投稿信息.csv')
test_userbehavior=pd.read_csv('G:/data/User_Identification/测试数据/用户行为信息.csv')

'''处理userinformation表'''
train1=pd.merge(train_userlabel,train_userinformation, on=['id'], how='left')
test_userlabel['label']=-1
test1=pd.merge(test_userlabel,test_userinformation, on=['id'], how='left')

gender_class_dict = {
    'male': 0,
    'female': 1,
    'unknown': 2
    }
train1['gender_str']=train1['gender_str'].map(gender_class_dict)
test1['gender_str']=test1['gender_str'].map(gender_class_dict)

#粉丝数与关注人数的差值
train1['fans_follow_difference'] = train1['fans_num_all'] - train1['follow_num_all']
test1['fans_follow_difference'] = test1['fans_num_all'] - test1['follow_num_all']
#粉丝数与关注人数的比例
train1['fans_follow_divide'] = train1['fans_num_all'] / (train1['follow_num_all']+1)
test1['fans_follow_divide'] = test1['fans_num_all'] / (test1['follow_num_all']+1)
#评论数与投稿数的差值
train1['comment_publish_difference'] = train1['server_comment_cnt_all'] - train1['publish_cnt_all']
test1['comment_publish_difference'] = test1['server_comment_cnt_all'] - test1['publish_cnt_all']
#评论数与投稿数的比例
train1['comment_publish_divide'] = train1['server_comment_cnt_all'] / (train1['publish_cnt_all']+1)
test1['comment_publish_divide'] = test1['server_comment_cnt_all'] / (test1['publish_cnt_all']+1)

train_signature_fastText_text = pd.read_csv('G:/data/User_Identification/fasttext/9552/train_signature_fastText_pre.csv')
test_signature_fastText_text = pd.read_csv('G:/data/User_Identification/fasttext/9552/test_signature_fastText_pre.csv')
train_signature_deeplearning_text = pd.read_csv('G:/data/User_Identification/deeplearning/pre/train_signature_deeplearning_pre.csv')
test_signature_deeplearning_text = pd.read_csv('G:/data/User_Identification/deeplearning/pre/test_signature_deeplearning_pre.csv')

train2=pd.merge(train1,train_signature_fastText_text[['id','fastText_pre']], on=['id'], how='left')
train2=pd.merge(train2,train_signature_deeplearning_text[['id','dpcnn_pre','rcnn_pre','rcnnattn_pre']], on=['id'], how='left')

test2=pd.merge(test1,test_signature_fastText_text[['id','fastText_pre']], on=['id'], how='left')
test2=pd.merge(test2,test_signature_deeplearning_text[['id','dpcnn_pre','rcnn_pre','rcnnattn_pre']], on=['id'], how='left')

#文本长度
def lenth_text(df):
    if isinstance(df,str):
        return len(df.split(','))
    else:
        return 0
train2['signature_lenth']=train2['signature'].apply(lenth_text)
test2['signature_lenth']=test2['signature'].apply(lenth_text)



'''处理userbehavior表'''
#播放完成占总播放次数的比例
train_userbehavior['video_finish_play_divide']=train_userbehavior['video_play_finish']/(train_userbehavior['video_play']+1)
test_userbehavior['video_finish_play_divide']=test_userbehavior['video_play_finish']/(test_userbehavior['video_play']+1)
#点击视频占总播放次数的比例
train_userbehavior['video_finish_click_divide']=train_userbehavior['click_video_play']/(train_userbehavior['video_play']+1)
test_userbehavior['video_finish_click_divide']=test_userbehavior['click_video_play']/(test_userbehavior['video_play']+1)
#点赞与点不喜欢的比例
train_userbehavior['dislike_like_divide']=train_userbehavior['dislike']/(train_userbehavior['like']+1)
test_userbehavior['dislike_like_divide']=test_userbehavior['dislike']/(test_userbehavior['like']+1)

#点赞、点不喜欢数、评论数、搜索数、分享数与播放时长的比例
train_userbehavior['like_time_divide']=train_userbehavior['like']/(train_userbehavior['play_time']+1)
train_userbehavior['dislike_time_divide']=train_userbehavior['dislike']/(train_userbehavior['play_time']+1)
train_userbehavior['comment_time_divide']=train_userbehavior['post_comment']/(train_userbehavior['play_time']+1)
train_userbehavior['search_time_divide']=train_userbehavior['search']/(train_userbehavior['play_time']+1)
train_userbehavior['share_time_divide']=train_userbehavior['share_video']/(train_userbehavior['play_time']+1)

test_userbehavior['like_time_divide']=test_userbehavior['like']/(test_userbehavior['play_time']+1)
test_userbehavior['dislike_time_divide']=test_userbehavior['dislike']/(test_userbehavior['play_time']+1)
test_userbehavior['comment_time_divide']=test_userbehavior['post_comment']/(test_userbehavior['play_time']+1)
test_userbehavior['search_time_divide']=test_userbehavior['search']/(test_userbehavior['play_time']+1)
test_userbehavior['share_time_divide']=test_userbehavior['share_video']/(test_userbehavior['play_time']+1)

#下滑与上滑的比例
train_userbehavior['slide_down_up_divide']=train_userbehavior['homepage_hot_slide_down']/(train_userbehavior['homepage_hot_slide_up']+1)
test_userbehavior['slide_down_up_divide']=test_userbehavior['homepage_hot_slide_down']/(test_userbehavior['homepage_hot_slide_up']+1)

train3=pd.merge(train2,train_userbehavior, on=['id'], how='left')
test3=pd.merge(test2,test_userbehavior, on=['id'], how='left')



'''处理usersubmission表'''
train_item_fastText_pre = pd.read_csv('G:/data/User_Identification/fasttext/train_item_fastText_pre.csv')
test_item_fastText_pre = pd.read_csv('G:/data/User_Identification/fasttext/test_item_fastText_pre.csv')

train_item_text = pd.read_csv('G:/data/User_Identification/deeplearning/pre/train_item_text.csv')
test_item_text = pd.read_csv('G:/data/User_Identification/deeplearning/pre/test_item_text.csv')

train_item_text['fastText_pre']=train_item_fastText_pre['fastText_pre']
test_item_text['fastText_pre']=test_item_fastText_pre['fastText_pre']

cross_num=['dpcnn_pre','rcnn_pre','rcnnattn_pre','fastText_pre']
cross_cat=['id']
# 定义交叉特征统计
def cross_cat_num(df, num_col, cat_col):
    for f1 in (cat_col):
        g = df.groupby(f1, as_index=False)
        for f2 in num_col:
            feat = g[f2].agg({
                '{}_{}_max'.format(f1, f2): 'max',
                '{}_{}_min'.format(f1, f2): 'min',
                '{}_{}_mean'.format(f1, f2): 'mean',
                '{}_{}_median'.format(f1, f2): 'median',
                '{}_{}_std'.format(f1, f2): 'std'
            })
            df = df.merge(feat, on=f1, how='left')
    return (df)

train_item_text = cross_cat_num(train_item_text, cross_num, cross_cat)
test_item_text = cross_cat_num(test_item_text, cross_num, cross_cat)

train_item_pre=train_item_text[['id','id_dpcnn_pre_max', 'id_dpcnn_pre_min',
       'id_dpcnn_pre_mean', 'id_dpcnn_pre_median', 'id_dpcnn_pre_std',
       'id_rcnn_pre_max', 'id_rcnn_pre_min', 'id_rcnn_pre_mean',
       'id_rcnn_pre_median', 'id_rcnn_pre_std', 'id_rcnnattn_pre_max',
       'id_rcnnattn_pre_min', 'id_rcnnattn_pre_mean', 'id_rcnnattn_pre_median',
       'id_rcnnattn_pre_std', 'id_fastText_pre_max', 'id_fastText_pre_min',
       'id_fastText_pre_mean', 'id_fastText_pre_median',
       'id_fastText_pre_std']].drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)

test_item_pre=test_item_text[['id','id_dpcnn_pre_max', 'id_dpcnn_pre_min',
       'id_dpcnn_pre_mean', 'id_dpcnn_pre_median', 'id_dpcnn_pre_std',
       'id_rcnn_pre_max', 'id_rcnn_pre_min', 'id_rcnn_pre_mean',
       'id_rcnn_pre_median', 'id_rcnn_pre_std', 'id_rcnnattn_pre_max',
       'id_rcnnattn_pre_min', 'id_rcnnattn_pre_mean', 'id_rcnnattn_pre_median',
       'id_rcnnattn_pre_std', 'id_fastText_pre_max', 'id_fastText_pre_min',
       'id_fastText_pre_mean', 'id_fastText_pre_median',
       'id_fastText_pre_std']].drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)

train4 = train_usersubmission.copy()
test4 = test_usersubmission.copy()

#处理时间戳数据
def timeStamp(df):
    timeArray = time.localtime(df)
    time_tran = time.strftime('%Y--%m--%d %H:%M:%S', timeArray)
    time_datetime = pd.to_datetime(time_tran)
    return time_datetime

train4['item_create_time'] = train4['item_create_time'].apply(timeStamp)
test4['item_create_time'] = test4['item_create_time'].apply(timeStamp)

#提交视频的时间段
train4['item_create_hour'] = train4['item_create_time'].apply(lambda x: x.hour)
test4['item_create_hour'] = test4['item_create_time'].apply(lambda x: x.hour)
#提交视频的日期
train4['item_create_day'] = train4['item_create_time'].apply(lambda x: x.day)
test4['item_create_day'] = test4['item_create_time'].apply(lambda x: x.day)
#提交视频的月份
train4['item_create_month'] = train4['item_create_time'].apply(lambda x: x.month)
test4['item_create_month'] = test4['item_create_time'].apply(lambda x: x.month)

#对province做映射
province_dict = {
       '贵州':1, '河南':2, '江苏':3, '江西':4, '四川':5, '广东':6, '山东':7, '浙江':8, '安徽':9, '湖南':10, '云南':11,
       '北京':12, '黑龙江':13, '重庆':14, '广西':14, '河北':16, '天津':17, '新疆':18, '山西':19, '湖北':20, '辽宁':21, '上海':22,
       '福建':23, '甘肃':24, '内蒙古':25, '吉林':26, '陕西':27, '宁夏':28, '青海':29, '海南':30, '西藏':31,
       '香港':32, '澳门':33, '台湾':34}

train4['item_province_cn']=train4['item_province_cn'].map(province_dict)
test4['item_province_cn']=test4['item_province_cn'].map(province_dict)
train4['item_province_cn']=train4['item_province_cn'].fillna(0)
test4['item_province_cn']=test4['item_province_cn'].fillna(0)
train4['item_province_cn']=train4['item_province_cn'].astype(int)
test4['item_province_cn']=test4['item_province_cn'].astype(int)

#对提交时段进行分箱
def time_segments(df):
    if (df>=0)&(df<3):
        return 0
    elif (df>=3)&(df<6):
        return 1
    elif (df>=6)&(df<9):
        return 2
    elif (df>=9)&(df<12):
        return 3
    elif (df>=12)&(df<15):
        return 4
    elif (df>=15)&(df<18):
        return 5
    elif (df>=18)&(df<21):
        return 6
    elif df>=21:
        return 7
train4['time_segments']=train4['item_create_hour'].apply(time_segments)
test4['time_segments']=test4['item_create_hour'].apply(time_segments)

#文本长度
def lenth_text(df):
    if isinstance(df,str):
        return len(df.split(','))
    else:
        return 0
train4['item_title_lenth']=train4['item_title'].apply(lenth_text)
test4['item_title_lenth']=test4['item_title'].apply(lenth_text)

def Users_characteristics(df):
    for f_pair in [['id','item_province_cn']]:#用户省份是否变更
        ### n unique、熵
        df = df.merge(df.groupby(f_pair[0], as_index=False)[f_pair[1]].agg({
            '{}_{}_nunique'.format(f_pair[0], f_pair[1]): 'nunique',
            '{}_{}_ent'.format(f_pair[0], f_pair[1]): lambda x: entropy(x.value_counts() / x.shape[0]),
            '{}_{}_mode'.format(f_pair[0], f_pair[1]): lambda x: x.value_counts().index[0]
        }), on=f_pair[0], how='left')

    return (df)
#用户购买产品特点
train4 = Users_characteristics(train4)
test4 = Users_characteristics(test4)


# 分组提取用户特征
def group_processing(df):
    for n in df.id.unique().tolist():
        group = df[df['id'] == n]

        # 提交了几天
        submit_day_change = len(group['item_create_day'].unique())

        # 提交次数
        submit_times = group['item_create_day'].count()
        df.loc[df['id'] == n, 'submit_times'] = submit_times

        # 每天提交频率
        submit_frequency = submit_times / submit_day_change
        df.loc[df['id'] == n, 'submit_frequency'] = submit_frequency

        # 提交期间的跨度
        item_create_dayspan = (group['item_create_time'].max() - group['item_create_time'].min()).days

        # 提交期间的频率
        submit_dayspan_frequency = submit_times / (item_create_dayspan + 1)
        df.loc[df['id'] == n, 'submit_dayspan_frequency'] = submit_dayspan_frequency

        # 更换视频标题的频率
        title_replace = len(group['item_title'].unique())
        df.loc[df['id'] == n, 'title_replace_frequency'] = title_replace / submit_times

        # 带poi的视频频率
        poi_times = len(group['poi_name'].unique())
        df.loc[df['id'] == n, 'poi_frequency'] = (poi_times - 1) / submit_times

    return df


train4 = group_processing(train4)
test4 = group_processing(test4)

train4['submit_times']=train4['submit_times'].astype('int')
test4['submit_times']=test4['submit_times'].astype('int')


def group_time_segments(df):
    for n in df.id.unique().tolist():
        group = df[df['id'] == n]
        submit_times = group['submit_times'].mean()
        time_segments_0_frequency = 0
        time_segments_1_frequency = 0
        time_segments_2_frequency = 0
        time_segments_3_frequency = 0
        time_segments_4_frequency = 0
        time_segments_5_frequency = 0
        time_segments_6_frequency = 0
        time_segments_7_frequency = 0

        item_hour_list = group['time_segments'].tolist()

        for i in item_hour_list:
            if i == 0:
                time_segments_0_frequency += 1
            elif i == 1:
                time_segments_1_frequency += 1
            elif i == 2:
                time_segments_2_frequency += 1
            elif i == 3:
                time_segments_3_frequency += 1
            elif i == 4:
                time_segments_4_frequency += 1
            elif i == 5:
                time_segments_5_frequency += 1
            elif i == 6:
                time_segments_6_frequency += 1
            elif i == 7:
                time_segments_7_frequency += 1

        df.loc[df['id'] == n, 'time_segments_0_frequency'] = time_segments_0_frequency / submit_times
        df.loc[df['id'] == n, 'time_segments_1_frequency'] = time_segments_1_frequency / submit_times
        df.loc[df['id'] == n, 'time_segments_2_frequency'] = time_segments_2_frequency / submit_times
        df.loc[df['id'] == n, 'time_segments_3_frequency'] = time_segments_3_frequency / submit_times
        df.loc[df['id'] == n, 'time_segments_4_frequency'] = time_segments_4_frequency / submit_times
        df.loc[df['id'] == n, 'time_segments_5_frequency'] = time_segments_5_frequency / submit_times
        df.loc[df['id'] == n, 'time_segments_6_frequency'] = time_segments_6_frequency / submit_times
        df.loc[df['id'] == n, 'time_segments_7_frequency'] = time_segments_7_frequency / submit_times

    return df


train4 = group_time_segments(train4)
test4 = group_time_segments(test4)

train5 = train4.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)
test5 = test4.drop_duplicates(subset=['id'], keep='first').reset_index(drop=True)

train5=train5.drop(['item_title','poi_name','item_province_cn','item_create_time','item_create_hour','item_create_day',
                    'item_create_month'], axis=1)
test5=test5.drop(['item_title','poi_name','item_province_cn','item_create_time','item_create_hour','item_create_day',
                    'item_create_month'], axis=1)

train6=pd.merge(train3,train5, on=['id'], how='left')
test6=pd.merge(test3,test5, on=['id'], how='left')
train7=pd.merge(train6,train_item_pre, on=['id'], how='left')
test7=pd.merge(test6,test_item_pre,on=['id'], how='left')

X = train7.drop(['label','id','feed_request','signature','create_time','fastText_pre','rcnn_pre','rcnnattn_pre',
                'id_rcnn_pre_max', 'id_rcnn_pre_min', 'id_rcnn_pre_mean',
                   'id_rcnn_pre_median', 'id_rcnn_pre_std', 'id_rcnnattn_pre_max',
                   'id_rcnnattn_pre_min', 'id_rcnnattn_pre_mean', 'id_rcnnattn_pre_median',
                   'id_rcnnattn_pre_std', 'id_fastText_pre_max', 'id_fastText_pre_min',
                   'id_fastText_pre_mean', 'id_fastText_pre_median',
                   'id_fastText_pre_std'], axis=1)
y = train7['label']
x_test = test7.drop(['label','id','feed_request','signature','create_time','fastText_pre','rcnn_pre','rcnnattn_pre',
                'id_rcnn_pre_max', 'id_rcnn_pre_min', 'id_rcnn_pre_mean',
                   'id_rcnn_pre_median', 'id_rcnn_pre_std', 'id_rcnnattn_pre_max',
                   'id_rcnnattn_pre_min', 'id_rcnnattn_pre_mean', 'id_rcnnattn_pre_median',
                   'id_rcnnattn_pre_std', 'id_fastText_pre_max', 'id_fastText_pre_min',
                   'id_fastText_pre_mean', 'id_fastText_pre_median',
                   'id_fastText_pre_std'], axis=1)


def plotroc(train_y, train_pred, test_y, val_pred):
    lw = 2
    ##train
    fpr, tpr, thresholds = roc_curve(train_y.values, train_pred, pos_label=1.0)
    train_auc_value = roc_auc_score(train_y.values, train_pred)
    ##valid
    fpr, tpr, thresholds = roc_curve(test_y.values, val_pred, pos_label=1.0)
    valid_auc_value = roc_auc_score(test_y.values, val_pred)

    return train_auc_value, valid_auc_value


def cv_lgb_model(clf, train_x, train_y, test_x, clf_name='lgb'):
    folds = 5
    seed = 42
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    train_oof = np.zeros(train_x.shape[0])
    test_oof = np.zeros(test_x.shape[0])

    categorical_feature = ['gender_str', 'id_item_province_cn_mode']
    cv_scores = []
    offline_score = []
    output_preds = 0
    feature_importance_df = pd.DataFrame()

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i + 1)))
        trn_x, trn_y = train_x.iloc[train_index], train_y[train_index]
        val_x, val_y = train_x.iloc[valid_index], train_y[valid_index]

        train_matrix = clf.Dataset(trn_x, label=trn_y)
        valid_matrix = clf.Dataset(val_x, label=val_y)

        params = {
            'boosting_type': 'gbdt',
            'objective': 'binary',
            'metric': {'auc'},  # 评估函数'auc','average_precision'
            'max_depth': 9,
            'num_leaves': 300,  # 叶子节点数
            'min_data_in_leaf': 50,
            'learning_rate': 0.05,  # 学习速率
            'feature_fraction': 0.7,  # 建树的特征选择比例
            'bagging_fraction': 0.7,  # 建树的样本采样比例
            'reg_alpha': 3.5,
            'reg_lambda': 2.6,
            'random_state': 42,
            'bagging_freq': 3,
            'verbose': -1  # <0 显示致命的, =0 显示错误 (警告), >0 显示信息
        }
        model = clf.train(params, train_matrix, 5000, valid_sets=[train_matrix, valid_matrix],
                          categorical_feature=categorical_feature,
                          verbose_eval=200, early_stopping_rounds=500)
        train_pred = model.predict(trn_x, num_iteration=model.best_iteration)
        val_pred = model.predict(val_x, num_iteration=model.best_iteration)
        test_pred = model.predict(test_x, num_iteration=model.best_iteration)
        train_oof[valid_index] = val_pred
        test_oof += test_pred / kf.n_splits
        cv_scores.append(roc_auc_score(val_y, val_pred))
        print(cv_scores)

        plt.figure(figsize=(12, 6))
        lgb.plot_importance(model, max_num_features=20, importance_type='gain')
        plt.title("Featurertances")
        plt.show()

        # 绘制roc曲线
        train_auc_value, valid_auc_value = plotroc(trn_y, train_pred, val_y, val_pred)
        print('train_auc:{},valid_auc{}'.format(train_auc_value, valid_auc_value))
        offline_score.append(valid_auc_value)
        print(offline_score)
        output_preds += test_pred / folds

        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = model.feature_name()
        fold_importance_df["importance"] = model.feature_importance(importance_type='gain')
        fold_importance_df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    print('all_auc:', roc_auc_score(train_y.values, train_oof))
    print(np.mean(offline_score))
    print('OOF-MEAN-AUC:%.6f, OOF-STD-AUC:%.6f' % (np.mean(offline_score), np.std(offline_score)))

    feature_sorted = feature_importance_df.groupby(['Feature'])['importance'].mean().sort_values(ascending=False)
    top_100_features = feature_importance_df.groupby(['Feature'])['importance'].mean().sort_values(
        ascending=False).head(100)

    return train_oof, test_oof, feature_sorted, top_100_features

lgb_train, lgb_test,feature_sorted, top_100_features = cv_lgb_model(lgb, X, y, x_test)

def sel(prob):
    prob_list=[]
    for pre in prob:
        if pre <= 0.88:
            prob_list.append(0)
        else:
            prob_list.append(1)
    prob_np=np.array(prob_list)
    return prob_np

def f_score(y_pre,y_true):
    y_pre=sel(y_pre)
    pre=pd.DataFrame(y_pre)
    pre['label_pre']=y_pre
    pre['label_true']=y_true

    true_positive=0
    false_positive=0
    false_negative =0
    true_negative=0

    for (i, r1) in pre.iterrows():
        pred_label = int(r1['label_pre'])
        truth_label = int(r1['label_true'])

        if pred_label == 1 and truth_label == 1:
            true_positive += 1
        elif pred_label == 1 and truth_label == 0:
            false_positive += 1
        elif pred_label == 0 and truth_label == 1:
            false_negative += 1
        elif pred_label == 0 and truth_label == 0:
            true_negative += 1

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
    accuracy = (true_positive+true_negative)/62433
    #f_score = 3 * precision * recall / (2 * precision + recall)
    f_score = (1+0.3 ** 2) * precision * recall / (0.3 **2 * precision + recall)
    print('本次预测结果：A={};P={}; R={}; F={}'.format(accuracy,precision, recall, f_score))

f_score(lgb_train,y)