import xgboost as xgb
import numpy as np
import pandas as pd
import catboost
from catboost import *
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
data = pd.read_csv('telecom-customer/Telecom_customer_churn.csv')


def prepare(raw, target, drop_first=True, make_na_col=True):
    dummy = pd.get_dummies(raw, columns=target,
            drop_first=drop_first,
            dummy_na=make_na_col)

    return dummy

data_ready = prepare(data, target=data.columns[data.dtypes.values=='object'])

data_ready.fillna(value=0,inplace=True)
data_ready = data_ready.drop(['churn', 'Customer_ID'], axis=1)

y = data['churn']


print(data.shape)
print(data_ready.shape)


xTrain, xVal, yTrain, yVal = train_test_split(data_ready,y,train_size=0.7, random_state=0)

xVal, xTest, yVal, yTest = train_test_split(xVal, yVal, train_size=0.5, random_state=0)

features = list(data_ready.columns)
#print(features)


xgbParams = {'max_depth': 3,
        'eta':0.01,
        'silent':0,
        'eval_metric':'auc',
        'subsample':0.8,
        'colsample_bytree':0.8,
        'objective':'binary:logistic',
        'seed':0}

dtrain = xgb.DMatrix(xTrain, yTrain, feature_names=xTrain.columns.values)

dVal = xgb.DMatrix(xVal, yVal, feature_names=xVal.columns.values)

dTest = xgb.DMatrix(xTest, yTest, feature_names=xTest.columns.values)

evals = [(dtrain,'train'), (dVal, 'eval')]

xgbModel = xgb.train(params = xgbParams,
        dtrain = dtrain,
        num_boost_round = 2000,
        verbose_eval = 50,
        early_stopping_rounds = 500,
        evals = evals,
        maximize = True)

xgbModel.save_model('xgModel.mdl')

#feat_imp = pd.DataFrame(list(xgbModel.get_fscore().items()),
        #columns=['feature', 'importance']).sort_values('importance', ascending=False)


#xgPreds = xgbModel.predict(dVal)
#print(xgPreds.shape)


#'''
catModel = CatBoostClassifier(iterations = 2000,
        learning_rate=0.1,
        task_type = "GPU",
        eval_metric='Accuracy' )

print('start fitting')
catModel.fit(xTrain, yTrain,
        eval_set=(xVal, yVal),
        verbose=50)

catModel.save_model('catModel.cbm')


'''
catPreds = model.predict_proba(xVal)

trainPreds = np.zeros((len(catPreds),2))

trainPreds[:,0] = xgPreds
trainPreds[:,1] = np.max(catPreds, axis=1)


logReg = LogisticRegression()
logReg.fit(trainPreds,yVal)

svm = SVC(probability=True)
svm.fit(trainPreds, yVal)


catPredsTest = model.predict_proba(xTest)
xgPredsTest = xgbModel.predict(dTest)

testPreds = np.zeros((len(catPredsTest),2))

testPreds[:,0] = xgPredsTest
testPreds[:,1] = np.max(catPredsTest,axis=1)


print('logAcc:',logReg.score(testPreds,yTest))

logPreds = logReg.predict_proba(testPreds)[::,1]

print('logAUC:',roc_auc_score(yTest,logPreds))


print('svmAcc: ', svm.score(testPreds, yTest))
svmPreds = svm.predict_proba(testPreds)[::,1]
print('svmAUC: ', roc_auc_score(yTest,svmPreds))
'''
