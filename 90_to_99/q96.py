#https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
#http://www.adeveloperdiary.com/data-science/deep-learning/nlp/machine-translation-using-attention-with-pytorch/
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle
import copy
from torch.utils.tensorboard import SummaryWriter

SOS_token = 0
EOS_token = 1
BATCH_SIZE = 256
TEACHER_FORCING_RATIO = 0.8
MAX_LEN = 51
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

with open("drive/MyDrive/nlp100_95/encoded_ja_train.txt", "rb") as fr_1:
    input_lang = pickle.load(fr_1)
with open("drive/MyDrive/nlp100_95/encoded_en_train.txt", "rb") as fr_2:
    output_lang = pickle.load(fr_2)

with open("drive/MyDrive/nlp100_95/encoded_ja_dev.txt", "rb") as fr_3:
    valid_input_lang = pickle.load(fr_3)
with open("drive/MyDrive/nlp100_95/encoded_en_dev.txt", "rb") as fr_4:
    valid_output_lang = pickle.load(fr_4)

input_oov_idx = 2
output_oov_idx = 2
input_padding_idx = 7999
output_padding_idx = 7999
train_data_num = len(input_lang)
valid_data_num = len(valid_input_lang)

def pad_list(list, pad_len, pad_idx):
    temp = list
    temp.append(1) #EOS
    while len(temp) < pad_len:
        temp.append(pad_idx)
    return temp[:pad_len]

def tensorsFromPair(batch_size):
    train_batch_num = train_data_num//batch_size
    temp_l = []
    for batch in range(train_batch_num):
        input_l = []
        target_l = []
        maximum_input_train_length = max([len(input_lang[batch*batch_size+i]) for i in range(batch_size)])+1
        maximum_output_train_length = max([len(output_lang[batch*batch_size+i]) for i in range(batch_size)])+1
        for i in range(batch_size): #ここもしかしたら最後の128個以下が無視されとるかも
            padded_input = pad_list(input_lang[batch*batch_size+i], min(MAX_LEN, maximum_input_train_length), input_padding_idx)
            padded_target = pad_list(output_lang[batch*batch_size+i], min(MAX_LEN, maximum_output_train_length), output_padding_idx)
            input_l.append(padded_input)
            target_l.append(padded_target)
        temp_l.append([torch.tensor(input_l).unsqueeze(2), torch.tensor(target_l).unsqueeze(2)])
    return temp_l, train_batch_num

def tensorsFromPair_valid(batch_size):
    valid_batch_num = valid_data_num//batch_size
    temp_l = []
    for batch in range(valid_batch_num):
        input_l = []
        target_l = []
        maximum_input_train_length = max([len(valid_input_lang[batch*batch_size+i]) for i in range(batch_size)])+1
        maximum_output_train_length = max([len(valid_output_lang[batch*batch_size+i]) for i in range(batch_size)])+1
        for i in range(batch_size): #ここもしかしたら最後の128個以下が無視されとるかも
            padded_input = pad_list(valid_input_lang[batch*batch_size+i], min(MAX_LEN, maximum_input_train_length), input_padding_idx)
            padded_target = pad_list(valid_output_lang[batch*batch_size+i], min(MAX_LEN, maximum_output_train_length), output_padding_idx)
            input_l.append(padded_input)
            target_l.append(padded_target)
        temp_l.append([torch.tensor(input_l).unsqueeze(2), torch.tensor(target_l).unsqueeze(2)])
    return temp_l, valid_batch_num

train_tensors, train_batch_num = tensorsFromPair(BATCH_SIZE) #[(日本語のバッチ1, 英語のバッチ1), (日本語のバッチ2, 英語のバッチ2),...] バッチの形は(バッチサイズ*テンソルの長さ*1)
valid_tensors, valid_batch_num = tensorsFromPair(BATCH_SIZE)

class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, word_embedding_dim, hidden_size):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=input_padding_idx)
        self.lstm = nn.LSTM(input_size = word_embedding_dim, hidden_size = hidden_size, batch_first = True)
        self.hidden_size = hidden_size

    def forward(self, raw_input):
        embedded_input = self.word_embedding(raw_input).squeeze(2) #ここのraw_inputは単語のidで構成されたテンソル (batch_size, seq_len, 1) -> (batch_size, seq_len, word_embedding_dim)
        batch_size = embedded_input.shape[0]
        initial_hidden_state = torch.zeros(1, batch_size, self.hidden_size)
        initial_cell_state = torch.zeros(1, batch_size, self.hidden_size)
        output, (h_n, c_n) = self.lstm(embedded_input, (initial_hidden_state, initial_cell_state))
        return output, h_n # output -> (batch_size, seq_len, hidden_size),     h_n(final hidden state) -> (1, batch_size, hidden_size)

class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.scoring_function = nn.Bilinear(in1_features = decoder_hidden_dim, in2_features = encoder_hidden_dim, out_features = 1, bias = False) #https://github.com/tensorflow/nmt
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, encoder_outputs, decoder_hidden_state, attention_mask):
        batch_size, seq_len, hidden_size = encoder_outputs.shape
        decoder_hidden_state = decoder_hidden_state.squeeze(0).unsqueeze(1).expand(batch_size, seq_len, hidden_size) #(1, batch_size, hidden_size) -> (batch_size, seq_len, hidden_size)
        attention_score = self.scoring_function(decoder_hidden_state, encoder_outputs) #encoder_outputs -> (batch_size, seq_len, hidden_size) #(batch_size, seq_len, 1)
        attention_score = attention_score.masked_fill(attention_mask, -float('inf'))
        attention_weight = self.softmax(attention_score) # attention_weight -> (batch_size, seq_len, 1)
        return attention_weight

class OneStepDecoder(nn.Module):
    def __init__(self, word_embedding_dim, encoder_hidden_dim, decoder_hidden_dim, vocab_size, attention):
        super().__init__()
        self.vocab_size = vocab_size
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim, padding_idx=output_padding_idx)
        self.lstm = nn.LSTM(input_size = word_embedding_dim + encoder_hidden_dim, hidden_size = decoder_hidden_dim, batch_first = True)
        self.attention = attention
        self.fc1 = nn.Linear(decoder_hidden_dim + encoder_hidden_dim + word_embedding_dim, vocab_size)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, predicted_token, decoder_hidden_state, encoder_hidden_states, decoder_cell_state, attention_mask):
        embedded_input = self.word_embedding(predicted_token) #(batch_size, 1, word_embedding_dim)
        attention_weight = self.attention(encoder_hidden_states, decoder_hidden_state, attention_mask) #(batch_size, seq_len, 1)
        attention_weight = attention_weight.transpose(1, 2) #(batch_size, 1, seq_len)
        context_vector = torch.bmm(attention_weight, encoder_hidden_states) #(batch_size, 1, seq_len)*(batch_size, seq_len, encoder_hidden_dim) -> (batch_size, 1, encoder_hidden_dim)
        #文脈ベクトル、前の時間ステップの出力、現在のデコーダーの隠れ層をlstmの入力にするので、まずは最初の二つを結合する
        lstm_input = torch.cat([embedded_input, context_vector], dim = 2) #(batch_size, 1, word_embedding_dim + encoder_hidden_dim)
        output, (next_hidden_state, next_cell_state) = self.lstm(lstm_input, (decoder_hidden_state, decoder_cell_state)) #(batch_size, 1, decoder_hidden_dim), (batch_size, 1, decoder_hidden_dim)
        #次の隠れ層が計算できたので、output（隠れ層）を使ってトークンの確率分布を計算
        #fc1の引数:(batch_size, decoder_hidden_dim+encoder_hidden_dim+word_embedding_dim)
        distributional_prob = self.fc1(torch.cat([output.squeeze(1), context_vector.squeeze(1), embedded_input.squeeze(1)], dim=1))
        return distributional_prob, next_hidden_state, next_cell_state #distributional_prob -> (batch_size, vocab_size)

class LSTMDecoderwithAttention(nn.Module):
    def __init__(self, one_step_decoder):
        super().__init__()
        self.one_step_decoder = one_step_decoder

    def forward(self, target, encoder_hidden_states, decoder_hidden_state, attention_mask, teacher_forcing_ratio):
        batch_size = target.shape[0]
        target_len = target.shape[1]
        target = target.transpose(0,1) #(seq_len, batch_size, 1)
        outputs = torch.zeros(target_len, batch_size, self.one_step_decoder.vocab_size).to(device) #ここ形注意 各時間ステップごとの、バッチサイズ分の予測を格納するので時間（出力の長さ）が先頭にくる
        input = torch.zeros(batch_size, 1, dtype = torch.long) #SOSのidは0 ここlongにしないとfloatだぞって怒られる
        decoder_cell_state = torch.zeros(1, batch_size, decoder_hidden_state.shape[2])

        for time_step in range(0, target_len): #バッチ全体でteacher forcingを使う/使わないを統一した方がいいのかしら　でもそれはそれでteacher forcingを使うバッチ・使わないバッチでlossの差がめっちゃ出たりしそう
            distributional_prob, decoder_hidden_state, decoder_cell_state = self.one_step_decoder(input, decoder_hidden_state, encoder_hidden_states, decoder_cell_state, attention_mask)
            outputs[time_step] = distributional_prob
            top_1 = distributional_prob.argmax(1)
            teacher_force = (random.random() < teacher_forcing_ratio) #パターン2
            input = target[time_step] if teacher_force else top_1.unsqueeze(1)
        return outputs

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, raw_input, target, attention_mask, teacher_forcing_ratio):
        encoder_hidden_states, final_hidden_state = self.encoder(raw_input)
        decoder_outputs = self.decoder(target, encoder_hidden_states, final_hidden_state, attention_mask, teacher_forcing_ratio)
        return decoder_outputs

def create_model(src_vocab_size, dst_vocab_size):
    word_embedding_dim = 300
    encoder_hidden_dim = 256
    decoder_hidden_dim = 256

    attention_model = Attention(encoder_hidden_dim, decoder_hidden_dim)
    encoder = LSTMEncoder(src_vocab_size, word_embedding_dim, encoder_hidden_dim)
    one_step_decoder = OneStepDecoder(word_embedding_dim, encoder_hidden_dim, decoder_hidden_dim, dst_vocab_size, attention_model)
    decoder = LSTMDecoderwithAttention(one_step_decoder)
    model = EncoderDecoder(encoder, decoder)
    model = model.to(device)

    optimizer = optim.AdamW(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index = output_padding_idx)

    return model, optimizer, criterion


def train(train_tensors, epoch_num, batch_size):
    model, optimizer, criterion = create_model(8000, 8000)
    train_batch_idx_list = [i for i in range(train_batch_num)]
    valid_batch_idx_list = [i for i in range(valid_batch_num)]
    train_masks = [train_tensors[i][0].eq(input_padding_idx) for i in range(train_batch_num)]
    valid_masks = [valid_tensors[i][0].eq(input_padding_idx) for i in range(valid_batch_num)]
    checkpoint_epochs = [epoch_num//3, (epoch_num//3)*2, epoch_num]

    for epoch in range(1, epoch_num+1):
        model.train()
        random.shuffle(train_batch_idx_list)
        counter = 0
        for id in train_batch_idx_list:
            source_data = train_tensors[id][0]
            target_data = train_tensors[id][1]
            model.zero_grad()
            optimizer.zero_grad()
            if epoch <= epoch_num//3:
              model_output = model(source_data, target_data, train_masks[id], 1.0) #target_len, batch_size, vocab_size
            else:
              model_output = model(source_data, target_data, train_masks[id], TEACHER_FORCING_RATIO)
            model_output = model_output.transpose(0, 1)
            model_output = model_output.reshape(-1, model_output.shape[2])
            target_data = target_data.reshape(-1)
            loss = criterion(model_output, target_data)
            loss.backward()
            optimizer.step()
            counter += 1
            if counter%100 == 0:
                print(f"epoch {epoch}")
                print(f"{counter} / {train_batch_num} done")
        print(f"epoch {epoch} done")

        with torch.no_grad():
            model.eval()
            train_loss = 0
            for id in train_batch_idx_list:
                source_data = train_tensors[id][0]
                target_data = train_tensors[id][1]
                model_output = model(source_data, target_data, train_masks[id], 0.0)
                model_output = model_output.transpose(0, 1)
                model_output = model_output.reshape(-1, model_output.shape[2])
                target_data = target_data.reshape(-1)
                loss = criterion(model_output, target_data)
                train_loss += loss.item()
            writer.add_scalar('loss/train_loss', train_loss/len(train_batch_idx_list), epoch)

            valid_loss = 0
            for id in valid_batch_idx_list:
                source_data = valid_tensors[id][0]
                target_data = valid_tensors[id][1]
                model_output = model(source_data, target_data, valid_masks[id], 0.0)
                model_output = model_output.transpose(0, 1)
                model_output = model_output.reshape(-1, model_output.shape[2])
                target_data = target_data.reshape(-1)
                loss = criterion(model_output, target_data)
                valid_loss += loss.item()
            writer.add_scalar('loss/valid_loss', valid_loss/len(valid_batch_idx_list), epoch)

writer = SummaryWriter(log_dir="./logs/exp1_tfr0.8")
train(train_tensors, 6, BATCH_SIZE)
writer.close()

#Colabで以下を実行
#%load_ext tensorboard
#%tensorboard --logdir=./drive/MyDrive/logs
