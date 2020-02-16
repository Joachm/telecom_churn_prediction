# telecom_churn_prediction
using DNN, catBoost, and xgBoost

download and read about the data set from:
https://www.kaggle.com/abhinav89/telecom-customer

1. run preluNetwork.py and trainBoosters.py to train the models used
The models are trained on 70 % of the data and validated on 15 %.

Note: it might take a while to train the neural network. moreTraining.py might be used to split the training up in different sessions for convenience.

2. run allModel.py to get final predictions from the ensemble of the three models.
A logistic regression classifier is trained on the predictions that the models make on the validation set. It is then tested on the final 15 % of the data set.
AUC should be > 0.71.
At the time of writing the best reported AUC on the kaggle page for this data set is 0.686 (Feb 16 2020).
