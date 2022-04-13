from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle

logreg_model = pickle.load(open("preserved_model.sav", 'rb'))

def calc_accuracy(data_type):
    with open(data_type+".txt", "r") as f:
        raw_data = f.readlines()
        y = [raw_data[i][0] for i in range(len(raw_data))]
        X = np.loadtxt(data_type+'.feature.txt')
        print(f"accuracy of {data_type} data: {logreg_model.score(X, y)}")

calc_accuracy("train")
calc_accuracy("test")
