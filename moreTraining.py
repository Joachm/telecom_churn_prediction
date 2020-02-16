from keras.models import load_model
from keras.callbacks import ModelCheckpoint
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv('telecom-customer/Telecom_customer_churn.csv')


def prepare(raw, target, drop_first=True, make_na_col=True):
    dummy = pd.get_dummies(raw, columns=target,
            drop_first=drop_first,
            dummy_na=make_na_col)

    return dummy

data_ready = prepare(data, target=data.columns[data.dtypes.values=='object'])

data_ready.fillna(value=data_ready.median(axis=0),inplace=True)
data_ready = data_ready.drop(['churn', 'Customer_ID'], axis=1)

y = data['churn']


print(data.shape)
print(data_ready.shape)


xTrain, xVal, yTrain, yVal = train_test_split(data_ready,y,train_size=0.7, random_state=0)

xVal, xTest, yVal, yTest = train_test_split(xVal, yVal, train_size=0.5, random_state=0)

filepath = 'prelu_model3.hdf5'
checkpoint = ModelCheckpoint(filepath,
        monitor='val_acc',
        verbose=1,
        save_best_only=True,
        mode='max')


model = load_model('prelu_model2.hdf5')

model.fit(xTrain, yTrain,
        epochs=2000,
        batch_size=128,
        validation_data=(xVal, yVal),
        verbose=1,
        callbacks=[checkpoint])
