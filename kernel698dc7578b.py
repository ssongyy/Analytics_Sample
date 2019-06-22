# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import os
print(os.listdir("../input"))
# Any results you write to the current directory are saved as output.
traindata_set = pd.read_csv('../input/train.csv')
test_set = pd.read_csv('../input/test.csv')
test_set
traindata_set.shape
test_set.shape
traindata_set.isnull().sum()
test_set.isnull().sum()
traindata_set.info
traindata_set.describe()

traindata_set['target'].value_counts()

target02=traindata_set[traindata_set['target']==0].iloc[:,2]
target12=traindata_set[traindata_set['target']==1].iloc[:,2]
target03=traindata_set[traindata_set['target']==0].iloc[:,3]
target13=traindata_set[traindata_set['target']==1].iloc[:,3]
fig=plt.figure(figsize=(12,8))
plt.title('1/0 distribution')
target02.hist(alpha=0.7,bins=30,label='0')
target12.hist(alpha=0.7,bins=30,label='1')
target03.hist(alpha=0.7,bins=30,label='0')
target13.hist(alpha=0.7,bins=30,label='1')
plt.legend(loc='upper right')

del traindata_set['ID_code']

traindata_set.columns
# Shuffle the Dataset.
shuffled_df = traindata_set.sample(frac=1,random_state=4)
# Put all the fraud class in a separate dataset.
tar1_df = shuffled_df.loc[shuffled_df['target'] == 1]
#Randomly select 492 observations from the non-fraud (majority class)
tar0_df = shuffled_df.loc[shuffled_df['target'] == 0].sample(n= 30098,random_state=42)
# Concatenate both dataframes again
normalized_df = pd.concat([tar1_df, tar0_df])
normalized_df['target'].values
#plot the dataset after the undersampling
plt.figure(figsize=(8, 8))
sns.countplot(x=normalized_df['target'].values)
plt.title('Balanced Classes')
plt.show()

X=normalized_df.iloc[:,1:]
y=normalized_df.iloc[:,0]

#feature selection
from sklearn import feature_selection
def feature_sele(n):
    select = feature_selection.SelectKBest(k=n)
    selected_features=select.fit(X,y)
    indices_selected=selected_features.get_support(indices=True)
    colnames=[traindata_set.columns[i] for i in indices_selected]
    colnames.remove('target')
    return(colnames)
    
#Model Building
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

#use seleceted dataset
def datasplit(data,colnames):
    xx=DataFrame(data,columns=colnames)
    return(xx)
xx1=datasplit(X,feature_sele(50))
xx2=datasplit(X,feature_sele(100))
xx3=datasplit(X,feature_sele(150))
xx4=datasplit(X,feature_sele(190))

#train data and test data split
train_x1,test_x1,train_y1,test_y1 = train_test_split(xx1,y,test_size=0.3,random_state=42)
train_x2,test_x2,train_y2,test_y2 = train_test_split(xx2,y,test_size=0.3,random_state=42)
train_x3,test_x3,train_y3,test_y3 = train_test_split(xx3,y,test_size=0.3,random_state=42)
train_x4,test_x4,train_y4,test_y4 = train_test_split(xx4,y,test_size=0.3,random_state=42)
print(train_x1,train_y1,test_x1,test_y1)
#build model
fore=RandomForestClassifier(n_estimators=100)
xgbm=xgb.XGBRegressor(
            learning_rate=0.1,
            max_depth=5,
            n_estimators=100,
            objective= 'binary:logistic',
            random_state=34) 
            
# build prediction function
def predict(m,train_x,train_y,test,test_y):
    m.fit(train_x,train_y)
    prediction = m.predict(test)
    return([test_y,prediction])
#Accuracy and Classification report
def Acu(y_test,pred):
    y_pred = [round(value) for value in pred]
    accuracy= accuracy_score(y_test,y_pred)
    report = classification_report(y_test,y_pred)
    print('Accuracy: '+ str(accuracy))
    print('report:'+str(report))
    
# Cross Validation for stability of the model    
from sklearn.model_selection import KFold, cross_val_score
def evaluation(model,x):
     k_fold = KFold(n_splits=5, shuffle=True, random_state=None)
     if model=='for':
         model=RandomForestClassifier(n_estimators=100)
     if model=='xgb':
         model=xgb.XGBRegressor(
            learning_rate=0.1,
            max_depth=5,
            n_estimators=100,
            objective= 'binary:logistic',
            random_state=34)
     cross_score = cross_val_score(model,x,y,cv=k_fold, n_jobs=1)
     print('Average Accuracy = ' + str(np.mean(cross_score)))
     print('Fold Accuracies = ' + str(cross_score))
# when we choose 50 features and use randomforest
#for_result=predict(fore,train_x1,train_y1,test_x1,test_y1)
#Acu(for_result[0],for_result[1])
#evaluation(fore,xx1)
# when we choose 100 features and use randomforest
for_result_1=predict(fore,train_x2,train_y2,test_x2,test_y2)
#Acu(for_result_1[0],for_result_1[1])
#evaluation(fore,xx2)
# when we choose 150 features and use randomforest
for_result_2=predict(fore,train_x3,train_y3,test_x3,test_y3)
#Acu(for_result_2[0],for_result_2[1])
#evaluation(fore,xx3)

#when we choose 190 features and use randomforest
#evaluation(fore,xx4)
for_result_3=predict(fore,train_x4,train_y4,test_x4,test_y4)
#Acu(for_result_3[0],for_result_3[1])
# when we choose 150 features and use xgboost
xgb_result=predict(xgbm,train_x3,train_y3,test_x3,test_y3)
#Acu(xgb_result[0],xgb_result[1])

#when we choose 190 features and use xgboost
xgb_result_1=predict(xgbm,train_x4,train_y4,test_x4,test_y4)
#Acu(xgb_result_1[0],xgb_result_1[1])

#Evaluation of model performance
from sklearn.metrics import roc_auc_score, roc_curve
print(roc_auc_score(for_result_2[0],for_result_2[1]))
print(roc_auc_score(xgb_result[0],xgb_result[1]))
print(roc_auc_score(for_result_3[0],for_result_3[1]))
print(roc_auc_score(xgb_result_1[0],xgb_result_1[1]))


fpr_for_150, tp_for_150, _ = roc_curve(for_result_2[0],for_result_2[1])
fpr_xgb_150, tp_xgb_150, _ = roc_curve(xgb_result[0],xgb_result[1])
fpr_for_190, tp_for_190, _ = roc_curve(for_result_3[0],for_result_3[1])
fpr_xgb_190, tp_xgb_190, _ = roc_curve(xgb_result_1[0],xgb_result_1[1])

plt.figure(figsize=(8, 8))
lw = 2

plt.plot(fpr_for_150, tp_for_150, color='purple', lw=lw, label='For-150')
plt.plot(fpr_xgb_150, tp_xgb_150, color='green', lw=lw, label='Xgb-150')
plt.plot(fpr_for_190, tp_for_190, color='orange', lw=lw, label='For-190')
plt.plot(fpr_xgb_190, tp_xgb_190, color='pink', lw=lw, label='Xgb-190')

plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

# Therefore, we use xgboost with 190 features as our final model:
re_tar=[]
new=datasplit(test_set,feature_sele(190))
result=xgb_result_1=predict(xgbm,train_x4,train_y4,new,test_y4)
my_submission = pd.DataFrame({'ID_code': test_set['ID_code'], 'target': result[1]})
my_submission.to_csv('submission.csv', index=False)



