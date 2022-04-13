from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
from matplotlib import pyplot as plt

with open("train.txt", "r") as f:
    raw_data = f.readlines()
    train_label = [raw_data[i][0] for i in range(len(raw_data))]
    train_features = np.loadtxt('train.feature.txt')
with open("valid.txt", "r") as f:
    raw_data = f.readlines()
    valid_label = [raw_data[i][0] for i in range(len(raw_data))]
    valid_features = np.loadtxt('valid.feature.txt')
with open("test.txt", "r") as f:
    raw_data = f.readlines()
    test_label = [raw_data[i][0] for i in range(len(raw_data))]
    test_features = np.loadtxt('test.feature.txt')

train_accuracy = []
valid_accuracy = []
test_accuracy = []
c_value = [10**(-3), 5*(10**(-3)), 10**(-2), 5*(10**(-2)) ,10**(-1), 5*(10**(-1)), 1.0, 5.0, 10.0]

for i in range(len(c_value)):
    logreg_model = LogisticRegression(C = c_value[i]).fit(train_features, train_label)
    print("training "+str(i)+" done")
    train_accuracy.append(logreg_model.score(train_features, train_label))
    valid_accuracy.append(logreg_model.score(valid_features, valid_label))
    test_accuracy.append(logreg_model.score(test_features, test_label))

plt.plot(np.array(c_value), np.array(train_accuracy), label = "train")
plt.plot(np.array(c_value), np.array(valid_accuracy), label = "valid")
plt.plot(np.array(c_value), np.array(test_accuracy), label = "test")
plt.xscale("log")
plt.xlabel("C")
plt.ylabel("accuracy")
plt.legend()
plt.show()
