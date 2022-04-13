import torch
import pickle
import torch.nn as nn

with open("word_dict.pkl", "rb") as tf:
    word_dict = pickle.load(tf)

vocab_size = max(word_dict.values())
embedding_dimension = 300
hidden_layer_dimension = 50


class MyRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size+1, embedding_dimension, padding_idx=0) #適当な埋め込み表現に変換。+1はid=0の分(padding)。
        self.rnn = nn.RNN(input_size = embedding_dimension, hidden_size = hidden_layer_dimension, batch_first = True)
        self.last_linear = nn.Linear(hidden_layer_dimension, 4, bias = False)
        self.softmax = nn.Softmax(dim = 1) #ソフトマックスは行単位（4つのカテゴリが入っている）について計算
    def forward(self, x):
        x = self.word_embedding(x) #入力xはtorch.tensor([1, 2, 3, 4, ...])みたいな単語のidの列。出力はサイズが(単語列の長さ*埋め込み次元数)のテンソルになる
        print(x)
        output_layers, final_hidden_state = self.rnn(x, None) #入力xと隠れ層の初期値(None)を受け取り、RNNの各時間ステップにおける隠れ層の状態と、その中でも一番最後の隠れ層の状態を出力
        print(output_layers[0])
        temp_y = output_layers[:, -1, :] #出力は(バッチサイズ*単語列の長さ*次元数)のテンソル型なので、一番最後の単語を読んだ後の隠れ層の状態をもらってくる
        y = self.softmax(self.last_linear(temp_y))
        return y

rnn = MyRNN()
L = torch.tensor([[1,0,3,3,4,0], [1,2,2,2,0,0], [1,2,1,3,3,2]])
print(rnn(L))
