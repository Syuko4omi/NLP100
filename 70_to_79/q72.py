import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

train_features = np.load("train_feature.npy")
train_labels = np.load("train_label.npy")
case_1 = torch.tensor([train_features[0]]).float() #ここfloatじゃないと怒られる（Tensor型に変換するときにテンソルの中の数値がtorch.double型になってしまうので、明示的にfloatに変換）
case_1_to_4 = torch.tensor(train_features[:4]).float()
target_1 = torch.tensor([train_labels[0]])
target_1_to_4 = torch.tensor(train_labels[:4])

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(300, 4, bias = False)

    def forward(self, x):
        x = F.softmax(self.fc1(x))
        return x

net = Net()
criterion = nn.CrossEntropyLoss()
"""
input_1 = case_1
output_1 = net(input_1)
loss = criterion(output_1, target_1)
print(loss)
loss.backward()
print(net.fc1.weight.grad) #biasはFalse
"""
input_1_to_4 = case_1_to_4
output_1_to_4 = net(input_1_to_4)
loss = criterion(output_1_to_4, target_1_to_4)
print(loss)
loss.backward()
print(net.fc1.weight.grad)
