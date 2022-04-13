#https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
#ここでは日本語->英語の翻訳を対象にする。Q91ではモデルの訓練を行う。
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle

class Lang:
    def __init__(self, language):
        self.language = language
        self.word2id = {}
        self.id2word = {0: "SOS", 1: "EOS"}
        self.word2count = {}
        self.num_of_words = 2

    def add_word_to_dict(self, word):
        if self.language == "ja":
            if word not in self.word2id:
                if word in ja_words_list:
                    self.word2id[word] = self.num_of_words
                    self.id2word[self.num_of_words] = word
                    self.word2count[word] = 1
                    self.num_of_words += 1
            else:
                self.word2count[word] += 1
        else:
            if word not in self.word2id:
                if word in en_words_list:
                    self.word2id[word] = self.num_of_words
                    self.id2word[self.num_of_words] = word
                    self.word2count[word] = 1
                    self.num_of_words += 1
            else:
                self.word2count[word] += 1

    def add_sentence_to_dict(self, sentence):
        for word in sentence.split():
            self.add_word_to_dict(word)

SOS_token = 0
EOS_token = 1
BATCH_SIZE = 128
EPOCH_NUM = 9
TEACHER_FORCING_RATIO = 0.9
MAX_LENGTH = 41
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
torch.set_default_tensor_type('torch.cuda.FloatTensor')

#データのロード
with open("drive/MyDrive/nlp100_91/input_lang.pkl", "rb") as fr_1:
    input_lang = pickle.load(fr_1)
with open("drive/MyDrive/nlp100_91/output_lang.pkl", "rb") as fr_2:
    output_lang = pickle.load(fr_2)
with open("drive/MyDrive/nlp100_91/train_pairs.txt", "rb") as fr_3:
    pairs = pickle.load(fr_3)
input_oov_idx = input_lang.num_of_words+1 #43222
output_oov_idx = output_lang.num_of_words+1 #48405
input_padding_idx = input_lang.num_of_words+2 #43223
output_padding_idx = output_lang.num_of_words+2 #48406

#日本語と英語の文を、単語に紐づけられた数字の列に置き換える
def indexesFromSentence(lang, sentence, io):
    temp_id_list = []
    for word in sentence.split():
        if word not in lang.word2id:
            if io == "input":
                temp_id_list.append(input_oov_idx)
            else:
                temp_id_list.append(output_oov_idx)
        else:
            temp_id_list.append(lang.word2id[word])
    return temp_id_list

def tensorFromSentence(lang, sentence, padding_length, io):
    indexes = indexesFromSentence(lang, sentence, io)
    indexes.append(EOS_token)
    if io == "input":
        while len(indexes) < padding_length:
            indexes.append(input_padding_idx)
    else:
        while len(indexes) < padding_length:
            indexes.append(output_padding_idx)
    return torch.tensor(indexes[:padding_length], dtype=torch.long).view(-1, 1)

def tensorsFromPair(pair, batch_size):
    batch_num = len(pair)//batch_size
    temp_l = []
    for batch in range(batch_num):
        input_l = []
        target_l = []
        maximum_input_train_length = max([len(pair[batch*batch_size+i][0].split()) for i in range(batch_size)])
        maximum_output_train_length = max([len(pair[batch*batch_size+i][1].split()) for i in range(batch_size)])
        for i in range(batch_size):
            input_tensor = tensorFromSentence(input_lang, pair[batch*batch_size+i][0], min(MAX_LENGTH, maximum_input_train_length+1), "input")
            target_tensor = tensorFromSentence(output_lang, pair[batch*batch_size+i][1], min(MAX_LENGTH, maximum_output_train_length+1), "output")
            input_l.append(input_tensor)
            target_l.append(target_tensor)
        temp_l.append([torch.stack(input_l, dim = 0), torch.stack(target_l, dim = 0)])
    return temp_l, batch_num

train_tensors, batch_num = tensorsFromPair(pairs, BATCH_SIZE) #[(日本語のバッチ1, 英語のバッチ1), (日本語のバッチ2, 英語のバッチ2),...] バッチの形は(バッチサイズ*テンソルの長さ*1)

class LSTMEncoder(nn.Module): #単語の列を受け取ってLSTMに通す。返り値は各時間ステップの隠れ層の状態と、最後の隠れ層の状態
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
        output, (h_n, c_n) = self.lstm(embedded_input, (initial_hidden_state, initial_cell_state)) #本当はh_nじゃなくて、実際にOOVじゃない単語を読んだところまでの隠れ層の状態を文脈ベクトルとして採用すべきだが、簡単のため最後の出力を用いる。
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
        #distributional_prob = self.softmax(self.fc1(torch.cat([output.squeeze(1), context_vector.squeeze(1), embedded_input.squeeze(1)], dim=1))) この行はめちゃくちゃ間違っているので注意。softmaxをかける必要はない
        output_of_fc = self.fc1(torch.cat([output.squeeze(1), context_vector.squeeze(1), embedded_input.squeeze(1)], dim=1))
        return output_of_fc, next_hidden_state, next_cell_state #distributional_prob -> (batch_size, vocab_size)

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

        for time_step in range(0, target_len):
            distributional_prob, decoder_hidden_state, decoder_cell_state = self.one_step_decoder(input, decoder_hidden_state, encoder_hidden_states, decoder_cell_state, attention_mask)
            outputs[time_step] = distributional_prob
            top_1 = distributional_prob.argmax(1)
            teacher_force = (random.random() < teacher_forcing_ratio)
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
    model, optimizer, criterion = create_model(input_lang.num_of_words+3, output_lang.num_of_words+3)
    batch_idx_list = [i for i in range(batch_num)]
    masks = [train_tensors[i][0].eq(input_padding_idx) for i in range(batch_num)]
    checkpoint_epochs = [epoch_num//3, (epoch_num//3)*2, epoch_num]

    for epoch in range(1, epoch_num+1):
        model.train()
        random.shuffle(batch_idx_list)
        counter = 0
        train_loss_ave_per_10 = 0
        for id in batch_idx_list:
            source_data = train_tensors[id][0]
            target_data = train_tensors[id][1]
            model.zero_grad()
            optimizer.zero_grad()
            if epoch <= epoch_num//3:
              model_output = model(source_data, target_data, masks[id], 1.0) #target_len, batch_size, vocab_size
            else:
              model_output = model(source_data, target_data, masks[id], TEACHER_FORCING_RATIO)
            model_output = model_output.transpose(0, 1)
            model_output = model_output.reshape(-1, model_output.shape[2])
            target_data = target_data.reshape(-1)
            loss = criterion(model_output, target_data)
            loss.backward()
            optimizer.step()
            train_loss_ave_per_10 += loss.item()
            counter += 1
            if counter%10 == 0:
                print(f"{counter} / {batch_num} done")
                print(f"average loss: {train_loss_ave_per_10/10}")
                train_loss_ave_per_10 = 0

        if epoch in checkpoint_epochs:
            checkpoint = {'model_state_dict': model.state_dict()}
            torch.save(checkpoint, f'trained_model_e{epoch}.pth')

train(train_tensors, EPOCH_NUM, BATCH_SIZE)
