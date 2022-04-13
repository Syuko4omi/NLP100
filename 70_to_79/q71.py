import numpy as np

def softmax(x):
    for i in range(len(x)):
        exp_sum = np.sum(np.exp(x[i]))
        x[i] = np.exp(x[i])/exp_sum
    return x

train_features = np.load("train_feature.npy")
train_labels = np.load("train_label.npy")

weight_matrix = np.random.rand(300, 4)

case_1 = [train_features[0]]
case_1_to_4 = train_features[:4]

print(softmax(np.dot(case_1, weight_matrix)))
print(softmax(np.dot(case_1_to_4, weight_matrix)))
