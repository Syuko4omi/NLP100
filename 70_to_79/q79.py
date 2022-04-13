import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_default_tensor_type('torch.cuda.FloatTensor')

epoch_num = 10000
learning_rate = 1

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
        self.fc1 = nn.Linear(300, 128, bias = True)
        self.fc2 = nn.Linear(128, 32, bias = True)
        self.fc3 = nn.Linear(32, 4, bias = True)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('using device:', device)
net = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr = learning_rate)

def training_step(batch_size):

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

training_step(2048)

outputs = net(train_case)
max_val, predicted_label = torch.max(outputs, 1)
correct_prediction_num = 0
for i in range(len(train_target)):
    if predicted_label[i] == train_target[i]:
        correct_prediction_num += 1
print(f"train accuracy: {correct_prediction_num/len(train_target)}")

outputs = net(test_case)
max_val, predicted_label = torch.max(outputs, 1)
correct_prediction_num = 0
for i in range(len(test_target)):
    if predicted_label[i] == test_target[i]:
        correct_prediction_num += 1
print(f"test accuracy: {correct_prediction_num/len(test_target)}")

#train accuracy:, 0.9473979782852864
#test accuracy:, 0.9116766467065869
