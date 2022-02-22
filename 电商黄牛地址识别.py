import shutil
import glob
import os
import gc
import seaborn as sns
import os
import datetime
import sys
import joblib
import toad
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import category_encoders as ce
from scipy.stats import entropy
from scipy import special, optimize
from scipy import spatial

from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import  RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.metrics import  f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans

from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings('ignore')

#读取数据
train=pd.read_csv('G:/data/电商黄牛地址识别/ec-scalpers/电商黄牛地址识别-题目数据/train.csv')
test=pd.read_csv('G:/data/电商黄牛地址识别/ec-scalpers/电商黄牛地址识别-题目数据/test.csv')

#合并数据&生成详细地址
data=pd.concat([train,test]).reset_index(drop=True)
data['detail_address']=data['post_province']+' '+data['post_city']+' '+data['post_town']+' '+data['post_detail']

# Word2Vec
def word_vector(df,size):
    # 地址列表
    raw_sentences = np.array(df).tolist()
    # 切分字符
    sentences= [s.encode('utf-8').split() for s in raw_sentences]
    # Word2Vec训练
    model = word2vec.Word2Vec(sentences,vector_size=size,min_count=0)
    # 单词数量
    print('单词数量',len(model.wv.index_to_key))
    #model.wv[2]
    # 词向量矩阵
    index2word_set = set(model.wv.index_to_key)
    return index2word_set,model
# 句子向量
def address_vector(sentence, model, num_features, index2word_set):
    words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        n_words += 1
        feature_vec = np.add(feature_vec, model.wv[int(word)])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec
# 地址向量
index2word_set,model = word_vector(data['detail_address'],100)
data['detail_address_vector']=data['detail_address'].apply(address_vector,args=(model, 100 ,index2word_set))
#单词数量 3923

#地址分类
def address_cate(data):
    for i in data['post_town'].unique():
        df=data[data['post_town']==i]
        if df['request_id'].count()==1:
            data.loc[data['post_town']==i,'address_cate_kmeans']=1
            #data.loc[data['post_town']==i,'address_cate_SpectralC']=1
            #data.loc[data['post_town']==i,'address_cate_DBSCAN']=1
        else:
            embedding = []
            for p in df['detail_address_vector']:
                embedding.append(p.tolist())
            n=len(embedding)
            #KMeans
            kmeans = KMeans(n_clusters=int(math.sqrt(n/2)),init='k-means++',random_state=42).fit(embedding)
            data.loc[data['post_town']==i,'address_cate_kmeans']=kmeans.labels_
            #谱聚类
            #SP = SpectralClustering(n_clusters=int(math.sqrt(n/2)),assign_labels='discretize',gamma=0.01,random_state=42).fit(embedding)
            #data.loc[data['post_town']==i,'address_cate_SpectralC']=SP.labels_
            #DBSCAN
            #DB = DBSCAN(eps=0.5,min_samples=5,metric='euclidean',algorithm='auto').fit(embedding)
            #data.loc[data['post_town']==i,'address_cate_DBSCAN']=DB.labels_
    return data
data=address_cate(data)

data['address_cluster_kmeans']=data['post_town']+' '+data['address_cate_kmeans'].astype('int').astype('str')
#data['address_cluster_SpectralC']=data['post_town']+' '+data['address_cate_SpectralC'].astype('int').astype('str')
#data['address_cluster_DBSCAN']=data['post_town']+' '+data['address_cate_DBSCAN'].astype('int').astype('str')




#label encoder
label_encoder_list=['post_province', 'post_city', 'post_town', 'post_detail','address_cluster_kmeans']
encoder = LabelEncoder()
encoded = data[label_encoder_list].apply(encoder.fit_transform)
df=pd.concat([data.drop(['post_province','post_city','post_town','post_detail','detail_address','detail_address_vector',
                         'address_cate_kmeans','address_cluster_kmeans'
                        ], axis=1),encoded],axis=1)

#频数编码（count encoding）
def count_encode(df, categorical_features):
    encoder = ce.count.CountEncoder(cols=categorical_features)
    df_count_encoded=encoder.fit_transform(df)[categorical_features]
    df = df.join(df_count_encoded.add_suffix('_count'))
    return df

#体现各分类订单数量
count_encoding_list=['product_id','user_id','product_1st_category','product_2nd_category','product_3rd_category','post_province','post_city','post_town']
df=count_encode(df, count_encoding_list)

#LabelCount编码
def labelcount_encode(df, categorical_features, ascending=False):
    df_ = pd.DataFrame()
    for cat_feature in categorical_features:
        cat_feature_value_counts = df[cat_feature].value_counts()
        value_counts_list = cat_feature_value_counts.index.tolist()
        if ascending:
            # 升序
            value_counts_range = list(reversed(range(len(cat_feature_value_counts))))
        else:
            # 降序
            value_counts_range = list(range(len(cat_feature_value_counts)))
        labelcount_dict = dict(zip(value_counts_list, value_counts_range))
        df_[cat_feature] = df[cat_feature].map(labelcount_dict)
    df_ = df_.add_suffix('_labelcount_encoded')
    if ascending:
        df_ = df_.add_suffix('_ascending')
    else:
        df_ = df_.add_suffix('_descending')
    df_ = df_.astype(int)
    df=pd.concat([df,df_],axis=1)
    print('LabelCount encoding: {}'.format(categorical_features),df_.shape)
    return df

#体现各分类订单数量排序
labelcount_encode_list=['product_id','product_1st_category','product_2nd_category','product_3rd_category','post_province','post_city','post_town']
df=labelcount_encode(df, labelcount_encode_list)




#统计各类用户特征
def Users_characteristics(df):
    for f_pair in [['user_id', 'post_detail'], ['post_detail', 'user_id']]:  # 用户地址是否变更
        ### n unique、熵
        df = df.merge(df.groupby(f_pair[0], as_index=False)[f_pair[1]].agg({
            '{}_{}_nunique'.format(f_pair[0], f_pair[1]): 'nunique',
        }), on=f_pair[0], how='left')

    for f_pair in [
        ['user_id', 'product_id'], ['user_id', 'product_1st_category'], ['user_id', 'product_2nd_category'],
        ['user_id', 'product_3rd_category']]:
        ###  #用户买几种商品，最多的商品是什么。买的商品种类与总购买数相比（是否专注于某种商品产生大量订单）
        df = df.merge(df.groupby(f_pair[0], as_index=False)[f_pair[1]].agg({
            '{}_{}_nunique'.format(f_pair[0], f_pair[1]): 'nunique',
            '{}_{}_mode'.format(f_pair[0], f_pair[1]): lambda x: x.value_counts().index[0],
            '{}_{}_ent'.format(f_pair[0], f_pair[1]): lambda x: entropy(x.value_counts() / x.shape[0])
        }), on=f_pair[0], how='left')
    df['product_id_nunique_divide_user_count'] = df['user_id_product_id_nunique'] / df['user_id_count']  ###该类别占总购买次数占比
    df['1st_category_divide_user_count'] = df['user_id_product_1st_category_nunique'] / df['user_id_count']
    df['2nd_category_divide_user_count'] = df['user_id_product_2nd_category_nunique'] / df['user_id_count']
    df['3rd_category_divide_user_count'] = df['user_id_product_3rd_category_nunique'] / df['user_id_count']

    ### 共现次数
    df['product_time'] = df.groupby(['product_id', 'user_id'])['request_id'].transform('count')  ### 用户买该商品的次数
    df['cate_time'] = df.groupby(['product_id', 'product_3rd_category'])['request_id'].transform('count')  ### 用户买该类别的次数
    df['product_time_divide_user_count'] = df['product_time'] / df['user_id_count']  ### 该商品占总购买次数占比
    df['cate_time_divide_user_count'] = df['product_time'] / df['user_id_count']  ### 该类别占总购买次数占比

    return (df)


# 用户购买产品特点
df = Users_characteristics(df)




#划分训练集测试集
train=df[df['label']!=-1].reset_index(drop=True)
test=df[df['label']==-1].reset_index(drop=True)

#划分训练集测试集
X = train.drop(['label','user_id','request_id','post_detail'], axis=1)
y = train['label']
x_test = test.drop(['label','user_id','request_id','post_detail'], axis=1)
#rules = test[['request_id','distance_nearest_scalper','scalper_count_9','scalper_count_95']]


def cv_cab1_model(train_x, train_y, test_x, clf_name='catboost'):
    folds = 6
    seed = 42
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    train_oof = np.zeros(train_x.shape[0])
    test_oof = np.zeros(test_x.shape[0])

    categorical_feature = ['product_id', 'post_province', 'post_city', 'post_town', 'product_1st_category',
                           'product_2nd_category', 'product_3rd_category',
                           'user_id_product_id_mode', 'user_id_product_1st_category_mode',
                           'user_id_product_2nd_category_mode',
                           'user_id_product_3rd_category_mode', 'address_cluster_kmeans']

    cv_scores = []
    offline_score = []
    output_preds = 0
    feature_importance_df = pd.DataFrame()

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i + 1)))
        trn_x, trn_y = train_x.iloc[train_index], train_y[train_index]
        val_x, val_y = train_x.iloc[valid_index], train_y[valid_index]

        catmodel = CatBoostClassifier(
            iterations=1000,
            depth=9,
            loss_function='CrossEntropy',
            eval_metric='AUC',
            cat_features=categorical_feature,
            learning_rate=0.05,
            random_seed=42,
            od_type='Iter')

        model = catmodel.fit(trn_x, trn_y, eval_set=(val_x, val_y), verbose=200, plot=True)

        train_pred = model.predict_proba(trn_x)[:, 1]
        val_pred = model.predict_proba(val_x)[:, 1]
        test_pred = model.predict_proba(test_x)[:, 1]
        train_oof[valid_index] = val_pred
        test_oof += test_pred / kf.n_splits
        cv_scores.append(roc_auc_score(val_y, val_pred))
        print(cv_scores)

        fold_importance_df = pd.DataFrame()
        fold_importance_df["Feature"] = model.feature_names_
        fold_importance_df["importance"] = model.feature_importances_
        fold_importance_df["fold"] = i + 1
        feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    print('all_auc:', roc_auc_score(train_y.values, train_oof))
    print(np.mean(offline_score))
    print('OOF-MEAN-AUC:%.6f, OOF-STD-AUC:%.6f' % (np.mean(offline_score), np.std(offline_score)))

    feature_sorted = feature_importance_df.groupby(['Feature'])['importance'].mean().sort_values(ascending=False)
    top_100_features = feature_importance_df.groupby(['Feature'])['importance'].mean().sort_values(
        ascending=False).head(100)

    return train_oof, test_oof, feature_sorted, top_100_features


#开始训练
catboost1_train, catboost1_test,catboost1_feature_sorted,catboost1_top100_feature = cv_cab1_model(X, y, x_test)



#计算f_score，选择合适的阈值
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
    accuracy = (true_positive+true_negative)/101887
    #f_score = 3 * precision * recall / (2 * precision + recall)
    f_score = (1+0.3 ** 2) * precision * recall / (0.3 **2 * precision + recall)
    print('本次预测结果：A={};P={}; R={}; F={}'.format(accuracy,precision, recall, f_score))
    print('预测结果：',f_score-0.012800167555781505)

f_score(catboost1_train,y)

test['label'] = catboost1_test
test['label'] = test['label'].apply(lambda x:1 if x>0.88 else 0).values
n=test['label']
#计算正负样本比例
a=test[test['label']==1]['label'].count()
b=test[test['label']==0]['label'].count()
c=a/b
print(a,b,c)

result=test.loc[:, ['request_id', 'label']]
result.to_csv('G:/data/电商黄牛地址识别/ec-scalpers/电商黄牛地址识别-题目数据/output.csv',index=None)

#stacking模型融合。将多结果线性回归
def cv_lr_model(train_x, train_y, test_x, clf_name='LR'):
    folds = 5
    seed = 42
    kf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)

    train_oof = np.zeros(train_x.shape[0])
    test_oof = np.zeros(test_x.shape[0])

    cv_scores = []
    offline_score = []
    output_preds = 0

    for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
        print('************************************ {} ************************************'.format(str(i + 1)))
        trn_x, trn_y = train_x.iloc[train_index], train_y[train_index]
        val_x, val_y = train_x.iloc[valid_index], train_y[valid_index]

        model = LinearRegression()
        model.fit(trn_x, trn_y)
        val_pred = model.predict(val_x)
        test_pred = model.predict(test_x)
        train_oof[valid_index] = val_pred
        test_oof += test_pred / kf.n_splits
        cv_scores.append(roc_auc_score(val_y, val_pred))
        print(cv_scores)

    return train_oof, test_oof