#https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
#ここでは日本語->英語の翻訳を対象にする
#モデルの訓練を行う
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, language):
        self.language = language
        self.word2id = {}
        self.id2word = {0: "SOS", 1: "EOS"}
        self.word2count = {}
        self.num_of_words = 2

    def add_word_to_dict(self, word):
        if word not in self.word2id:
            self.word2id[word] = self.num_of_words
            self.id2word[self.num_of_words] = word
            self.word2count[word] = 1
            self.num_of_words += 1
        else:
            self.word2count[word] += 1

    def add_sentence_to_dict(self, sentence):
        for word in sentence.split():
            self.add_word_to_dict(word)

def read_data(src_lang, dst_lang, data_type): #ここではまだlangの中身を作らない
    with open(f"kftt-data-1.0/data/tok/kyoto-{data_type}.{src_lang}", "r") as fr_1:
        lines_1 = fr_1.readlines()
        with open(f"kftt-data-1.0/data/tok/kyoto-{data_type}.{dst_lang}", "r") as fr_2:
            lines_2 = fr_2.readlines()
            pairs_of_src_and_dst = [[lines_1[i][:-1], lines_2[i][:-1]] for i in range(len(lines_1))]

            return Lang(src_lang), Lang(dst_lang), pairs_of_src_and_dst

def create_lang(src_lang, dst_lang, data_type): #langの中身を作る(trainのみ使用。testやdevでは使わない)
    input_lang, output_lang, pairs = read_data(src_lang, dst_lang, data_type)
    for pair in pairs:
        input_lang.add_sentence_to_dict(pair[0])
        output_lang.add_sentence_to_dict(pair[1])
    print("Counted words:")
    print(input_lang.language, input_lang.num_of_words)
    print(output_lang.language, output_lang.num_of_words)
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = create_lang('ja', 'en', "train.cln")
padding_idx = max((input_lang.num_of_words)+1, (output_lang.num_of_words)+1)
#print(random.choice(pairs))
#print(list(input_lang.word2count.values()).count(1)/len(list(input_lang.word2count.values())))
#print(list(output_lang.word2count.values()).count(1)/len(list(output_lang.word2count.values())))
#print((list(input_lang.word2count.values()).count(1)+list(input_lang.word2count.values()).count(2))/len(list(input_lang.word2count.values())))
t = list(output_lang.word2count.values())
for i in range(1, 11):
    print(sum(x<=i for x in t)/len(list(output_lang.word2count.values())))
counter = 0
temp = 0
while counter < 30:
    if t[temp] == 3:
        print(output_lang.id2word[temp])
        counter += 1
    temp += 1
t.sort(reverse = True)
#print(t[:100])
"""
def indexesFromSentence(lang, sentence):
    return [lang.word2id[word] for word in sentence.split()]

def tensorFromSentence(lang, sentence, padding_length):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    while len(indexes) < padding_length+1: #+1はeosの分
        indexes.append(padding_idx)
    return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

def tensorsFromPair(pair, batch_size):
    batch_num = len(pair)//batch_size
    temp_l = []
    for batch in range(batch_num):
        input_l = []
        target_l = []
        maximum_input_train_length = max([len(pair[batch*batch_size+i][0].split()) for i in range(batch_size)])
        maximum_output_train_length = max([len(pair[batch*batch_size+i][1].split()) for i in range(batch_size)])
        for i in range(batch_size): #ここもしかしたら最後の128個以下が無視されとるかも
            input_tensor = tensorFromSentence(input_lang, pair[batch*batch_size+i][0], maximum_input_train_length)
            target_tensor = tensorFromSentence(output_lang, pair[batch*batch_size+i][1], maximum_output_train_length)
            input_l.append(input_tensor)
            target_l.append(target_tensor)
        temp_l.append([torch.stack(input_l, dim = 0), torch.stack(target_l, dim = 0)])
    return temp_l, batch_num

train_tensors, batch_num = tensorsFromPair(pairs, batch_size = 128) #[(日本語のバッチ1, 英語のバッチ1), (日本語のバッチ2, 英語のバッチ2),...] バッチの形は(バッチサイズ*テンソルの長さ*1)
#print(train_tensors[0][0].shape)
#print(train_tensors[0][1].shape)

class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, word_embedding_dim, hidden_size): #word_embedding_dim == hidden_size(hidden state dimension)?
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim)
        self.lstm = nn.LSTM(input_size = word_embedding_dim, hidden_size = hidden_size, batch_first = True)
        self.hidden_size = hidden_size

    def forward(self, raw_input):
        embedded_input = self.word_embedding(raw_input).squeeze(2) #ここのraw_inputは単語のidで構成されたテンソル (batch_size, seq_len, 1) -> (batch_size, seq_len, word_embedding_dim)
        batch_size = embedded_input.shape[0]
        initial_hidden_state = torch.zeros(1, batch_size, self.hidden_size)
        initial_cell_state = torch.zeros(1, batch_size, self.hidden_size)
        output, (h_n, c_n) = self.lstm(embedded_input, (initial_hidden_state, initial_cell_state))
        # output -> (batch_size, seq_len, hidden_size)
        # h_n(final hidden state) -> (batch_size, 1, hidden_size)
        return output, h_n

class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.scoring_function = nn.Bilinear(in1_features = decoder_hidden_dim, in2_features = encoder_hidden_dim, out_features = 1, bias = False)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, encoder_outputs, decoder_hidden_state, attention_mask):
        batch_size, seq_len, hidden_size = encoder_outputs.shape #torch.Size([1, 128, 256])
        #copy_of_decoder_hidden_state = decoder_hidden_state.detach()
        #copy_of_decoder_hidden_state = copy_of_decoder_hidden_state.squeeze(0)
        #copy_of_decoder_hidden_state = copy_of_decoder_hidden_state.unsqueeze(1).expand(batch_size, seq_len, hidden_size) #(batch_size, hidden_size) -> (batch_size, seq_len, hidden_size)
        #attention_score = self.scoring_function(copy_of_decoder_hidden_state, encoder_outputs) #(batch_size, seq_len, 1)
        decoder_hidden_state = decoder_hidden_state.squeeze(0).unsqueeze(1).expand(batch_size, seq_len, hidden_size)
        attention_score = self.scoring_function(decoder_hidden_state, encoder_outputs)
        attention_score = attention_score.masked_fill(attention_mask, -float('inf'))
        attention_weight = self.softmax(attention_score) # attention_weight -> (batch_size, seq_len, 1)
        return attention_weight

class OneStepDecoder(nn.Module):
    def __init__(self, word_embedding_dim, encoder_hidden_dim, decoder_hidden_dim, vocab_size, attention):
        super().__init__()
        self.vocab_size = vocab_size
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim)
        self.lstm = nn.LSTM(input_size = word_embedding_dim + encoder_hidden_dim, hidden_size = decoder_hidden_dim, batch_first = True)
        self.attention = attention
        self.fc1 = nn.Linear(decoder_hidden_dim + encoder_hidden_dim + word_embedding_dim, vocab_size)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, predicted_token, decoder_hidden_state, encoder_hidden_states, decoder_cell_state, attention_mask):
        embedded_input = self.word_embedding(predicted_token) #(batch_size, 1, word_embedding_dim)
        attention_weight = self.attention(encoder_hidden_states, decoder_hidden_state, attention_mask) #(batch_size, seq_len, 1)
        attention_weight = attention_weight.transpose(1,2) #(batch_size, 1, seq_len)
        context_vector = torch.bmm(attention_weight, encoder_hidden_states) #(batch_size, 1, seq_len)*(batch_size, seq_len, encoder_hidden_dim) -> (batch_size, 1, encoder_hidden_dim)
        #文脈ベクトル、前の時間ステップの出力、現在のデコーダーの隠れ層をlstmの入力にするので、まずは最初の二つを結合する
        lstm_input = torch.cat([embedded_input, context_vector], dim = 2) #(batch_size, 1, word_embedding_dim + encoder_hidden_dim)
        output, (next_hidden_state, next_cell_state) = self.lstm(lstm_input, (decoder_hidden_state, decoder_cell_state)) #(batch_size, 1, decoder_hidden_dim), (batch_size, 1, decoder_hidden_dim)
        #次の隠れ層が計算できたので、output（隠れ層）を使ってトークンの確率分布を計算
        #fc1の引数:(batch_size, decoder_hidden_dim+encoder_hidden_dim+word_embedding_dim)
        predicted_token = self.softmax(self.fc1(torch.cat([output.squeeze(1), context_vector.squeeze(1), embedded_input.squeeze(1)], dim=1)))
        return predicted_token, next_hidden_state, next_cell_state #predicted token -> (batch_size, vocab_size)

class LSTMDecoderwithAttention(nn.Module):
    def __init__(self, one_step_decoder):
        super().__init__()
        self.one_step_decoder = one_step_decoder

    def forward(self, target, encoder_hidden_states, decoder_hidden_state, attention_mask, teacher_forcing_ratio=0.5):
        batch_size = target.shape[0]
        target_len = target.shape[1]
        outputs = torch.zeros(target_len, batch_size, self.one_step_decoder.vocab_size) #ここ形注意 各時間ステップごとの、バッチサイズ分の予測を格納するので時間（出力の長さ）が先頭にくる
        input = torch.zeros(batch_size, 1, dtype = torch.long) #SOSのidは0 ここlongにしないとfloatだぞって怒られる
        decoder_cell_state = torch.zeros(1, batch_size, decoder_hidden_state.shape[2])

        for time_step in range(0, target_len): #バッチ全体でteacher forcingを使う/使わないを統一した方がいいのかしら　でもそれはそれでteacher forcingを使うバッチ・使わないバッチでlossの差がめっちゃ出たりしそう
            predicted_token, decoder_hidden_state, decoder_cell_state = self.one_step_decoder(input, decoder_hidden_state, encoder_hidden_states, decoder_cell_state, attention_mask)
            outputs[time_step] = predicted_token
            teacher_force = (random.random() < teacher_forcing_ratio)
            top_1 = predicted_token.argmax(1)
            input = torch.tensor([target[i][time_step] for i in range(batch_size)]).unsqueeze(1) if teacher_force else top_1.unsqueeze(1)
            #print(input)
        return outputs

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, raw_input, target, attention_mask, teacher_forcing_ratio=0.5):
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

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index = padding_idx)

    return model, optimizer, criterion


def train(train_tensors, epoch_num, batch_size):
    aaaa = max(input_lang.num_of_words, output_lang.num_of_words)+10 #ここやばいので直す
    model, optimizer, criterion = create_model(aaaa, aaaa)
    batch_idx_list = [i for i in range(batch_num)]

    for epoch in range(1, epoch_num+1):
        model.train()
        random.shuffle(batch_idx_list)
        for id in batch_idx_list:
            source_data = train_tensors[id][0]
            target_data = train_tensors[id][1]
            optimizer.zero_grad()
            model_output = model(source_data, target_data, source_data.eq(padding_idx), teacher_forcing_ratio=0.5) #batch_size, target_len, vocab_size
            model_output = model_output.transpose(0, 1)
            model_output = model_output.transpose(1, 2)
            loss = criterion(model_output, target_data.squeeze(2))
            loss.backward()
            optimizer.step()
            print(loss)

train(train_tensors, 10, 128)
"""
