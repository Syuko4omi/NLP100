import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt

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

train_loss_list = []
train_accuracy_list = []
test_loss_list = []
test_accuracy_list = []

for epoch in range(epoch_num):
    net.zero_grad()
    optimizer.zero_grad()
    train_outputs = net(train_case)
    train_loss = criterion(train_outputs, train_target)
    train_loss_list.append(float(train_loss))
    test_outputs = net(test_case)
    test_loss = criterion(test_outputs, test_target)
    test_loss_list.append(float(test_loss))

    train_loss.backward()
    optimizer.step()
    max_val, predicted_label = torch.max(train_outputs, 1)
    correct_prediction_num = 0
    for i in range(len(train_target)):
        if predicted_label[i] == train_target[i]:
            correct_prediction_num += 1
    train_accuracy_list.append(correct_prediction_num/len(train_target))

    max_val, predicted_label = torch.max(test_outputs, 1)
    correct_prediction_num = 0
    for i in range(len(test_target)):
        if predicted_label[i] == test_target[i]:
            correct_prediction_num += 1
    test_accuracy_list.append(correct_prediction_num/len(test_target))

    if epoch%(epoch_num//5) == (epoch_num//5)-1:
        print(f"{epoch+1}: {train_loss}")
        torch.save(net.state_dict(), f"{epoch+1}_epoch.pt")

fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)
ax1.plot(np.array([i for i in range(epoch_num)]), np.array(train_loss_list), label = "train_loss")
ax1.plot(np.array([i for i in range(epoch_num)]), np.array(test_loss_list), label = "test_loss")
ax2.plot(np.array([i for i in range(epoch_num)]), np.array(train_accuracy_list), label = "train_accuracy")
ax2.plot(np.array([i for i in range(epoch_num)]), np.array(test_accuracy_list), label = "test_accuracy")
ax1.legend()
ax2.legend()
plt.show()
