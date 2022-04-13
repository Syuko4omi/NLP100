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

solver_list = ["newton-cg", "lbfgs", "liblinear", "sag"]
penalty_list = [["l2"], ["l2"], ["l1", "l2"], ["l2"]]
c_value = [10**(-1), 1.0, 10.0]

best_accuracy = 0.0
best_parameters = ["", "", None]
for i in range(len(solver_list)):
    for j in range(len(penalty_list[i])):
        for k in range(len(c_value)):
            logreg_model = LogisticRegression(max_iter = 1000, penalty = penalty_list[i][j], solver = solver_list[i], C = c_value[k]).fit(train_features, train_label)
            valid_score = logreg_model.score(valid_features, valid_label)
            if best_accuracy < valid_score:
                best_accuracy = valid_score
                best_parameters[0] = penalty_list[i][j]
                best_parameters[1] = solver_list[i]
                best_parameters[2] = c_value[k]
            print(valid_score, penalty_list[i][j], solver_list[i], c_value[k])

print("best model:", best_parameters)
best_logreg_model = LogisticRegression(max_iter = 1000, penalty = best_parameters[0], solver = best_parameters[1], C = best_parameters[2]).fit(train_features, train_label)
print("test accuracy:", best_logreg_model.score(test_features, test_label))
