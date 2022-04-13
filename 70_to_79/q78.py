import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
import time
torch.set_default_tensor_type('torch.cuda.FloatTensor')

epoch_num = 100
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('using device:', device)
net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = learning_rate)

def training_step(batch_size):
    start_time = time.time()

    for epoch in range(epoch_num):
        for iteration in range(len(train_case)//batch_size):
            train_batch = train_case[iteration*batch_size:(iteration+1)*batch_size]
            net.zero_grad()
            optimizer.zero_grad()
            train_outputs = net(train_batch)
            train_loss = criterion(train_outputs, train_target[iteration*batch_size:(iteration+1)*batch_size])

            train_loss.backward()
            optimizer.step()

        if epoch%(epoch_num//10) == (epoch_num//10)-1:
            print(f"epoch: {epoch+1}")

    print(f"batch size: {batch_size}")
    print(f"elapsed time per one epoch: {(time.time()-start_time)/epoch_num}")

training_step(128)
