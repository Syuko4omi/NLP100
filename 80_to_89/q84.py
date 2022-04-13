import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gensim
import pickle
import random

with open("word_dict.pkl", "rb") as tf:
    word_dict = pickle.load(tf)
pretrained_word_embedding = gensim.models.KeyedVectors.load_word2vec_format("../60_to_69/GoogleNews-vectors-negative300.bin", binary=True)
weights = pretrained_word_embedding.syn0
vocab_size = weights.shape[0]

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
                if word in pretrained_word_embedding:
                    temp.append(pretrained_word_embedding.vocab[word].index)
                else:
                    temp.append(0)
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
                if word in pretrained_word_embedding:
                    temp.append(pretrained_word_embedding.vocab[word].index)
                else:
                    temp.append(0)
        test_converted_titles.append(temp)
        maximum_test_title_length = max(maximum_test_title_length, len(temp))
for test_converted_title in test_converted_titles:
    while len(test_converted_title) < maximum_test_title_length:
        test_converted_title.append(0)
test_converted_titles = torch.tensor(test_converted_titles)

embedding_dimension = 300
hidden_layer_dimension = 50

class MyRNN(nn.Module):
    def __init__(self, activation_function):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size+1, embedding_dimension, padding_idx=0) #適当な埋め込み表現に変換。+1はid=0の分(padding)。
        self.word_embedding.weight = nn.Parameter(torch.from_numpy(weights))
        self.rnn = nn.RNN(input_size = embedding_dimension, hidden_size = hidden_layer_dimension, batch_first = True, nonlinearity = activation_function)
        self.last_linear = nn.Linear(hidden_layer_dimension, 4, bias = True)
        self.softmax = nn.Softmax(dim = 1) #ソフトマックスは行単位（4つのカテゴリが入っている）について計算

    def forward(self, x):
        x = self.word_embedding(x) #入力xはtorch.tensor([1, 2, 3, 4, ...])みたいな単語のidの列。出力はサイズが(単語列の長さ*埋め込み次元数)のテンソルになる
        output_layers, final_hidden_state = self.rnn(x, None) #入力xと隠れ層の初期値(None)を受け取り、RNNの各時間ステップにおける隠れ層の状態と、その中でも一番最後の隠れ層の状態を出力
        temp_y = output_layers[:, -1, :] #出力は(バッチサイズ*単語列の長さ*次元数)のテンソル型なので、一番最後の単語を読んだ後の隠れ層の状態をもらってくる
        y = self.last_linear(temp_y)
        #y = self.softmax(self.last_linear(temp_y))
        return y

def training_step(batch_size, learning_rate, epoch_num, activation_function, batch_shuffle):
    rnn = MyRNN(activation_function)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(rnn.parameters(), lr = learning_rate)
    batch_num = len(train_labels)//batch_size
    for epoch in range(epoch_num):

        temp_l = [i for i in range(batch_num)]
        if batch_shuffle:
            random.shuffle(temp_l)
        for iteration in temp_l:
            rnn.zero_grad()
            optimizer.zero_grad()
            train_outputs = rnn(train_converted_titles[iteration*batch_size:(iteration+1)*batch_size])
            train_loss = criterion(train_outputs, train_labels[iteration*batch_size:(iteration+1)*batch_size])
            train_loss.backward()
            optimizer.step()
        print(train_loss)

        if epoch%(epoch_num//10) == (epoch_num//10)-1:
            outputs = rnn(test_converted_titles)
            max_val, predicted_label = torch.max(outputs, 1)
            correct_prediction_num = 0
            for i in range(len(test_labels)):
                if predicted_label[i] == test_labels[i]:
                    correct_prediction_num += 1
            print(f"epoch: {epoch+1}")
            print(f"loss: {float(train_loss)}")
            print(f"test accuracy: {correct_prediction_num/len(test_labels)}")

#training_step(batch_size = 256, learning_rate = 0.01, epoch_num = 500, activation_function = "relu", batch_shuffle = True) #500itでloss0.9 test accuracy:70%
training_step(batch_size = 2048, learning_rate = 0.08, epoch_num = 500, activation_function = "relu", batch_shuffle = True)
