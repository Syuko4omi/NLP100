import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

epoch_num = 1000
learning_rate = 100

train_features = np.load("train_feature.npy")
train_labels = np.load("train_label.npy")
test_features = np.load("test_feature.npy")
test_labels = np.load("test_label.npy")
train_case = torch.tensor(train_features).float()
train_target = torch.tensor(train_labels)
test_case = torch.tensor(test_features).float()
test_target = torch.tensor(test_labels)

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(300, 4, bias = False)

    def forward(self, x):
        x = F.softmax(self.fc1(x))
        return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = learning_rate)

for epoch in range(epoch_num):
    net.zero_grad()
    optimizer.zero_grad()
    output = net(train_case)
    loss = criterion(output, train_target)
    loss.backward()
    optimizer.step()
    if epoch%(epoch_num//10) == (epoch_num//10)-1:
        print(f"{epoch+1}: {loss}")

outputs = net(train_case)
max_val, predicted_label = torch.max(outputs, 1)
correct_prediction_num = 0
for i in range(len(train_target)):
    if predicted_label[i] == train_target[i]:
        correct_prediction_num += 1
print(f"train accuracy:, {correct_prediction_num/len(train_target)}")

outputs = net(test_case)
max_val, predicted_label = torch.max(outputs, 1)
correct_prediction_num = 0
for i in range(len(test_target)):
    if predicted_label[i] == test_target[i]:
        correct_prediction_num += 1
print(f"test accuracy:, {correct_prediction_num/len(test_target)}")
