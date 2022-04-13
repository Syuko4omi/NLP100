import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

word_embedding_dim = 300
conved_feature_dim = 50

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embedding = nn.Embedding(100, word_embedding_dim, padding_idx=0)
        self.cnn = nn.Conv1d(in_channels = word_embedding_dim, out_channels = conved_feature_dim, kernel_size = 3)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size = 4)
        self.linear = nn.Linear(in_features = conved_feature_dim, out_features = 4, bias = True)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.word_embedding(x)
        x = x.transpose(2,1)
        output_cnn = self.relu(self.cnn(x))
        output_max_pool = self.max_pool(output_cnn)
        squeezed_output_max_pool = torch.squeeze(output_max_pool)
        output_linear = self.linear(squeezed_output_max_pool)
        output_softmax = self.softmax(output_linear)
        return output_softmax

model = MyCNN()
for_test = torch.tensor([[1,2,3,4,5,0],[2,4,2,1,5,3],[4,0,0,2,0,0],[8,1,2,0,5,0]])
print(model(for_test))
