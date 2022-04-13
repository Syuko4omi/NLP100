import torch
import torch.nn as nn
import numpy as np

class LSTMEncoder(nn.Module):
    def __init__(self, vocab_size, word_embedding_dim, hidden_size):
        super().__init__()
        self.word_embedding = nn.Embedding(vocab_size, word_embedding_dim)
        self.lstm = nn.LSTM(input_size = word_embedding_dim, hidden_size = hidden_size, batch_first = True)
        self.hidden_size = hidden_size

    def forward(self, raw_input):
        embedded_input = self.word_embedding(raw_input).squeeze(2)
        batch_size = embedded_input.shape[0]
        initial_hidden_state = torch.zeros(1, batch_size, self.hidden_size)
        initial_cell_state = torch.zeros(1, batch_size, self.hidden_size)
        output, (h_n, c_n) = self.lstm(embedded_input, (initial_hidden_state, initial_cell_state))
        return output, h_n

class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.scoring_function = nn.Bilinear(in1_features = decoder_hidden_dim, in2_features = encoder_hidden_dim, out_features = 1, bias = False)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, encoder_outputs, decoder_hidden_state, attention_mask):
        batch_size, seq_len, hidden_size = encoder_outputs.shape
        decoder_hidden_state = decoder_hidden_state.squeeze(0).unsqueeze(1).expand(batch_size, seq_len, hidden_size)
        attention_score = self.scoring_function(decoder_hidden_state, encoder_outputs)
        attention_score = attention_score.masked_fill(attention_mask, -float('inf'))
        attention_weight = self.softmax(attention_score)
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

    def forward(self, target, encoder_hidden_states, decoder_hidden_state, attention_mask):
        batch_size = target.shape[0]
        target_len = target.shape[1]
        outputs = torch.zeros(target_len, batch_size, self.one_step_decoder.vocab_size) #ここ形注意 各時間ステップごとの、バッチサイズ分の予測を格納するので時間（出力の長さ）が先頭にくる
        input = torch.zeros(batch_size, 1, dtype = torch.long) #SOSのidは0 ここlongにしないとfloatだぞって怒られる
        decoder_cell_state = torch.zeros(1, batch_size, decoder_hidden_state.shape[2])

        for time_step in range(0, target_len): #バッチ全体でteacher forcingを使う/使わないを統一した方がいいのかしら　でもそれはそれでteacher forcingを使うバッチ・使わないバッチでlossの差がめっちゃ出たりしそう
            predicted_token, decoder_hidden_state, decoder_cell_state = self.one_step_decoder(input, decoder_hidden_state, encoder_hidden_states, decoder_cell_state, attention_mask)
            outputs[time_step] = predicted_token
            top_1 = predicted_token.argmax(1)
            input = top_1.unsqueeze(1)
        return outputs

class EncoderDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, raw_input, target, attention_mask):
        encoder_hidden_states, final_hidden_state = self.encoder(raw_input)
        decoder_outputs = self.decoder(target, encoder_hidden_states, final_hidden_state, attention_mask)
        return decoder_outputs
