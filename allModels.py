from keras.models import load_model
from catboost import *
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import numpy as np
import pandas as pd

#### LOAD AND PREPARE DATA ####
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


xTrain, xVal, yTrain, yVal = train_test_split(data_ready,y,train_size=0.7)

xVal, xTest, yVal, yTest = train_test_split(xVal, yVal, train_size=0.5)

dVal = xgb.DMatrix(xVal, yVal, feature_names=xVal.columns.values)
dTest = xgb.DMatrix(xTest, yTest, feature_names=xTest.columns.values)

#### LOAD MODELS ####
catModel = CatBoostClassifier()
catModel.load_model('catModel.cbm')

xgModel = xgb.Booster()
xgModel.load_model('xgModel.mdl')

nnModel = load_model('prelu_model3.hdf5')


#### MAKE PREDICTIONS ON VALIDATION DATA ####
xgPreds = xgModel.predict(dVal)
catPreds = np.max(catModel.predict_proba(xVal), axis=1)
nnPreds = np.max(nnModel.predict_proba(xVal), axis=1)

valPreds = np.zeros((len(catPreds),3))
valPreds[:,0] = xgPreds
valPreds[:,1] = catPreds
valPreds[:,2] = nnPreds


#### FIT LOGISTIC REGRESSION ON PREDICTIONS
logistic = LogisticRegression()
logistic.fit(valPreds, yVal)


#### MAKE TEST PREDICTIONS ####

xgPreds2 = xgModel.predict(dTest)
catPreds2 = np.max(catModel.predict_proba(xTest), axis=1)
nnPreds2 = np.max(nnModel.predict_proba(xTest), axis=1)

testPreds = np.zeros((len(catPreds2),3))
testPreds[:,0] = xgPreds2
testPreds[:,1] = catPreds2
testPreds[:,2] = nnPreds2


#### FINAL TEST ####

print('Final accuracy',logistic.score(testPreds, yTest))
logPreds = logistic.predict_proba(testPreds)[::,1]
print('Final AUC: ', roc_auc_score(yTest, logPreds))
