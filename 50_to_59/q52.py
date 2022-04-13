from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle

with open("train.txt", "r") as fr_1:
    raw_train_data = fr_1.readlines()
    y = [raw_train_data[i][0] for i in range(len(raw_train_data))]
    X = np.loadtxt('train.feature.txt')
    logreg_model = LogisticRegression().fit(X, y)
    #print(logreg_model.score(X, y))
    pickle.dump(logreg_model, open("preserved_model.sav", "wb"))
