from sklearn.metrics import confusion_matrix
import pickle
import numpy as np

logreg_model = pickle.load(open("preserved_model.sav", 'rb'))

def make_conf_matrix(file_type):
    with open(file_type+".txt", "r") as f:
        raw_data = f.readlines()
        y = [raw_data[i][0] for i in range(len(raw_data))]
        X = np.loadtxt(file_type+".feature.txt")
        predictions = logreg_model.predict(X)
        cm = confusion_matrix(y, predictions)
        print(cm)

make_conf_matrix("train")
make_conf_matrix("test")
