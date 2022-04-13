import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

epoch_num = 1000

train_features = np.load("train_feature.npy")
train_labels = np.load("train_label.npy")
train_case = torch.tensor(train_features).float()
train_target = torch.tensor(train_labels).long()

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(300, 4, bias = False)

    def forward(self, x):
        x = F.softmax(self.fc1(x))
        return x

net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = 0.1)

for epoch in range(epoch_num):
    net.zero_grad()
    optimizer.zero_grad()
    output = net(train_case)
    loss = criterion(output, train_target)
    loss.backward()
    optimizer.step()
    if epoch%100 == 99:
        print(f"{epoch}: {loss}")
