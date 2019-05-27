# coding: utf-8
'''
@ liyue
@ 2018.11
'''
import pandas as pd, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import xgboost as xgb
import time
import feature
import pickle

#1.load data
df_all=pd.read_csv('../data/all_data_1019.csv')
df_train=df_all[df_all['label']!=-1]
y_train=(df_train["label"]).astype(int)
df_test=df_all[df_all['label']==-1].drop('label',axis=1)
test_id=df_test[['file_name']].copy()
column_name="code"

#2.feature extract
x_train_tf, x_test_tf = feature.get_feature_tf(df_all, df_train, df_test, y_train, column_name, save=True)
x_train_tfidf, x_test_tfidf = feature.get_feature_tfidf(df_all, df_train, df_test, y_train, column_name, save=True)

tf_path = './data_tf.pkl'
x_train_lda, x_test_lda = feature.get_feature_lda(tf_path, save=True)
tfidf_path = './data_tfidf.pkl'
x_train_lsa, x_test_lsa = feature.get_feature_lsa(tfidf_path, save=True)

f1 = open('./feat_lda.pkl', 'rb')
x_train_lda, y_train, x_test_lda = pickle.load(f1)
f1.close()
f2 = open('./feat_lsa.pkl', 'rb')
x_train_lsa, y_train, x_test_lsa = pickle.load(f2)
f2.close()

# featrue merge
x_train = np.concatenate((x_train_lda, x_train_lsa), axis=1)
x_test = np.concatenate((x_test_lda, x_test_lsa), axis=1)

# 3.XGBClassifier
clf =  xgb.XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=100)
clf.fit(x_train, y_train)
preds=clf.predict_proba(x_test)

# 4.result
preds=np.argmax(preds,axis=1)
test_pred=pd.DataFrame(preds)
test_pred.columns=["label"]
test_pred["判定结果"]=(test_pred["label"]).astype(int)

test_pred['判定结果'][test_pred['label']==0]='white'
test_pred['判定结果'][test_pred['label']==1]='black'

print(test_pred.shape)
print(test_id.shape)
test_pred["脚本文件名称"]=list(test_id["file_name"])
test_pred[["脚本文件名称","判定结果"]].to_csv('result.csv',index=None)
