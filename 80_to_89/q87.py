import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import random
from sklearn.metrics import classification_report

with open("word_dict.pkl", "rb") as tf:
    word_dict = pickle.load(tf)
train_labels = torch.tensor(np.load("train_label.npy"))
test_labels = torch.tensor(np.load("test_label.npy"))

train_converted_titles = []
maximum_train_title_length = 0
with open("train.txt", "r") as f_r:
    L = f_r.readlines()
    for i in range(len(L)):
        title = L[i][2:len(L)-1].split()
        temp = []
        for word in title:
            if word not in word_dict:
                temp.append(0)
            else:
                temp.append(word_dict[word])
        train_converted_titles.append(temp)
        maximum_train_title_length = max(maximum_train_title_length, len(temp))
for train_converted_title in train_converted_titles:
    while len(train_converted_title) < maximum_train_title_length:
        train_converted_title.append(0)
train_converted_titles = torch.tensor(train_converted_titles)

test_converted_titles = []
maximum_test_title_length = 0
with open("test.txt", "r") as f_r:
    L = f_r.readlines()
    for i in range(len(L)):
        title = L[i][2:len(L)-1].split()
        temp = []
        for word in title:
            if word not in word_dict:
                temp.append(0)
            else:
                temp.append(word_dict[word])
        test_converted_titles.append(temp)
        maximum_test_title_length = max(maximum_test_title_length, len(temp))
for test_converted_title in test_converted_titles:
    while len(test_converted_title) < maximum_test_title_length:
        test_converted_title.append(0)
test_converted_titles = torch.tensor(test_converted_titles)

vocab_size = max(word_dict.values())
word_embedding_dim = 300
conved_feature_dim = 50

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=0)
        self.cnn = nn.Conv1d(in_channels = word_embedding_dim, out_channels = conved_feature_dim, kernel_size = 3)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size = 15)
        self.linear = nn.Linear(in_features = conved_feature_dim, out_features = 4, bias = True)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.word_embedding(x)
        x = x.transpose(2,1) #次元がバッチサイズ*文の長さ*埋め込み次元数になっているが、cnnの入力にするために文の長さと埋め込み次元を交換する（これは問題ない操作）
        output_cnn = self.relu(self.cnn(x))
        output_max_pool = self.max_pool(output_cnn)
        squeezed_output_max_pool = torch.squeeze(output_max_pool) #バッチサイズ*文の長さ*各成分の次元数(1次元)->バッチサイズ*文の長さに
        output_linear = self.linear(squeezed_output_max_pool)
        #output_softmax = self.softmax(output_linear)
        #return output_softmax
        return output_linear


def training_step(batch_size, learning_rate, epoch_num, batch_shuffle):
    cnn = MyCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(cnn.parameters(), lr = learning_rate)
    batch_num = len(train_labels)//batch_size
    for epoch in range(epoch_num):

        temp_l = [i for i in range(batch_num)]
        if batch_shuffle:
            random.shuffle(temp_l)
        for iteration in temp_l:
            cnn.zero_grad()
            optimizer.zero_grad()
            train_outputs = cnn(train_converted_titles[iteration*batch_size:(iteration+1)*batch_size])
            train_loss = criterion(train_outputs, train_labels[iteration*batch_size:(iteration+1)*batch_size])
            train_loss.backward()
            optimizer.step()
        print(train_loss)

        if epoch%(epoch_num//10) == (epoch_num//10)-1:
            outputs = cnn(test_converted_titles)
            max_val, predicted_label = torch.max(outputs, 1)
            correct_prediction_num = 0
            for i in range(len(test_labels)):
                if predicted_label[i] == test_labels[i]:
                    correct_prediction_num += 1
            print(f"epoch: {epoch+1}")
            print(f"loss: {float(train_loss)}")
            print(f"test accuracy: {correct_prediction_num/len(test_labels)}")
            print(classification_report(test_labels, predicted_label))

training_step(batch_size = 128, learning_rate = 0.5, epoch_num = 50, batch_shuffle = True)
