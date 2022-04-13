from sklearn.metrics import classification_report
import pickle
import numpy as np

logreg_model = pickle.load(open("preserved_model.sav", 'rb'))

def calc_classification_report(file_type):
    with open(file_type+".txt", "r") as f:
        raw_data = f.readlines()
        y = [raw_data[i][0] for i in range(len(raw_data))]
        X = np.loadtxt(file_type+".feature.txt")
        predictions = logreg_model.predict(X)
        print(classification_report(y, predictions))

calc_classification_report("test")
