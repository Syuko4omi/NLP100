import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pickle
import random

batch_size = 256

with open("re_word_dict.pkl", "rb") as tf:
    word_dict = pickle.load(tf)
train_labels = torch.tensor(np.load("train_label.npy"))
test_labels = torch.tensor(np.load("test_label.npy"))
oov_id = max(word_dict.values())

train_converted_titles = []
train_titles_virtual_length = []
with open("train.txt", "r") as f_r_1:
    L = f_r_1.readlines()
    for i in range(len(L)):
        title = L[i][2:len(L)-1].split()
        temp = []
        for word in title:
            if word not in word_dict:
                temp.append(oov_id)
            else:
                temp.append(word_dict[word])
        train_converted_titles.append(temp)
#train_converted_titles.sort(key = lambda x: len(x)) #タイトルの長さでソート（同じバッチ中にあるタイトルは長さが近くなって、パディングが少なくなるように）
for i in range(len(train_labels)//batch_size):
    maximum_train_title_length = max(len(train_converted_titles[i*batch_size+j]) for j in range(batch_size))
    for j in range(batch_size):
        train_titles_virtual_length.append(len(train_converted_titles[i*batch_size+j]))
        while len(train_converted_titles[i*batch_size+j]) < maximum_train_title_length:
            train_converted_titles[i*batch_size+j].append(0)
#train_converted_titles = torch.tensor(train_converted_titles)

test_converted_titles = []
with open("test.txt", "r") as f_r_2:
    L = f_r_2.readlines()
    for i in range(len(L)):
        title = L[i][2:len(L)-1].split()
        temp = []
        for word in title:
            if word not in word_dict:
                temp.append(oov_id)
            else:
                temp.append(word_dict[word])
        test_converted_titles.append(temp)

embedding_dimension = 300
hidden_layer_dimension = 50

class MyRNN(nn.Module):
    def __init__(self, activation_function):
        super().__init__()
        self.word_embedding = nn.Embedding(oov_id+1, embedding_dimension, padding_idx=0) #適当な埋め込み表現に変換。+1はid=0の分(padding)。
        self.rnn = nn.RNN(input_size = embedding_dimension, hidden_size = hidden_layer_dimension, batch_first = True, nonlinearity = activation_function)
        self.last_linear = nn.Linear(hidden_layer_dimension, 4, bias = True)
        self.softmax = nn.Softmax(dim = 1) #ソフトマックスは行単位（4つのカテゴリが入っている）について計算
    def forward(self, x, v_len):
        x = self.word_embedding(x) #入力xはtorch.tensor([1, 2, 3, 4, ...])みたいな単語のidの列。出力はサイズが(単語列の長さ*埋め込み次元数)のテンソルになる
        output_layers, final_hidden_state = self.rnn(x, None) #入力xと隠れ層の初期値(None)を受け取り、RNNの各時間ステップにおける隠れ層の状態と、その中でも一番最後の隠れ層の状態を出力
        #temp_y = output_layers[:, -1, :] #出力は(バッチサイズ*単語列の長さ*次元数)のテンソル型なので、一番最後の単語を読んだ後の隠れ層の状態をもらってくる
        temp_y = [output_layers[:, max(0, int(virtual_len)-1), :] for virtual_len in v_len]
        y = self.last_linear(temp_y[0])
        return y

def training_step(learning_rate, epoch_num, activation_function, batch_shuffle):
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
            train_outputs = rnn(torch.tensor(train_converted_titles[iteration*batch_size:(iteration+1)*batch_size]), torch.tensor(train_titles_virtual_length[iteration*batch_size:(iteration+1)*batch_size]))
            train_loss = criterion(train_outputs, torch.tensor(train_labels[iteration*batch_size:(iteration+1)*batch_size]))
            train_loss.backward()
            optimizer.step()
        print(train_loss)

        if epoch%(epoch_num//10) == (epoch_num//10)-1:
            train_correct_prediction_num = 0
            oov_counter = 0
            for id in range(len(train_labels)):
                outputs = rnn(torch.tensor([train_converted_titles[id]]), torch.tensor([len(train_converted_titles[id])]))
                max_val, predicted_label = torch.max(outputs, 1)
                if predicted_label[0] == train_labels[id]:
                    train_correct_prediction_num += 1
                for j in range(len(train_converted_titles[id])):
                    if train_converted_titles[id][j] == oov_id:
                        oov_counter += 1
            print(f"epoch: {epoch+1}")
            print(f"loss: {float(train_loss)}")
            print(f"train accuracy: {train_correct_prediction_num/len(train_labels)}")
            #print(train_correct_prediction_num, len(train_labels))
            #print(f"ave oov train: {oov_counter/len(train_labels)}")

            oov_counter = 0
            test_correct_prediction_num = 0
            for id in range(len(test_labels)):
                #print([test_converted_titles[id]], [len(test_converted_titles[id])])
                outputs = rnn(torch.tensor([test_converted_titles[id]]), torch.tensor([len(test_converted_titles[id])]))
                max_val, predicted_label = torch.max(outputs, 1)
                if predicted_label[0] == test_labels[id]:
                    test_correct_prediction_num += 1
                for j in range(len(test_converted_titles[id])):
                    if test_converted_titles[id][j] == oov_id:
                        oov_counter += 1
            print(f"test accuracy: {test_correct_prediction_num/len(test_labels)}")
            #print(test_correct_prediction_num, len(test_labels))
            #print(f"ave oov test: {oov_counter/len(test_labels)}")

training_step(learning_rate = 0.01, epoch_num = 100, activation_function = "tanh", batch_shuffle = True) #100itでloss0.9 test accuracy:75%
