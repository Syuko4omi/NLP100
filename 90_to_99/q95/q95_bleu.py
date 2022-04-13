import torch
import torch.nn as nn
import numpy as np
import nltk
from nltk import bleu_score
import pickle
import sentencepiece as spm

from encoder_decoder_model import LSTMEncoder, Attention, OneStepDecoder, LSTMDecoderwithAttention, EncoderDecoder

def create_model_for_inference(src_vocab_size, dst_vocab_size):
    word_embedding_dim = 300
    encoder_hidden_dim = 256
    decoder_hidden_dim = 256

    attention_model = Attention(encoder_hidden_dim, decoder_hidden_dim)
    encoder = LSTMEncoder(src_vocab_size, word_embedding_dim, encoder_hidden_dim)
    one_step_decoder = OneStepDecoder(word_embedding_dim, encoder_hidden_dim, decoder_hidden_dim, dst_vocab_size, attention_model)
    decoder = LSTMDecoderwithAttention(one_step_decoder)
    model = EncoderDecoder(encoder, decoder)
    return model

with open("encoded_ja_train.txt", "rb") as fr_1:
    input_lang = pickle.load(fr_1)
with open("encoded_en_train.txt", "rb") as fr_2:
    output_lang = pickle.load(fr_2)

mt_model = create_model_for_inference(8000, 8000)
checkpoint_file = '../trained_model/q95_model/trained_model_e6.pth'
checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
mt_model.load_state_dict(checkpoint["model_state_dict"])
ja_model = spm.SentencePieceProcessor(model_file='ja_vocab.model')
en_model = spm.SentencePieceProcessor(model_file='en_vocab.model')

bleu_scores = []
test_pairs = []
with open("../kftt-data-1.0/data/orig/kyoto-test.ja", "r") as fr_1:
    with open("../kftt-data-1.0/data/orig/kyoto-test.en", "r") as fr_2:
        L_1 = fr_1.readlines()
        L_2 = fr_2.readlines()
        for i in range(len(L_1)):
            test_pairs.append([L_1[i][:len(L_1[i])-1], L_2[i][:len(L_2[i])-1]]) #改行除く

for pair in test_pairs[1:]: #最初のデータは除く
    source = ja_model.encode([pair[0]])
    source[0].append(1)
    target = en_model.encode([pair[1]])
    src_tensor = torch.tensor(source)
    attention_mask = torch.ones(1, src_tensor.shape[1], 1)

    encoder_hidden_states, encoder_last_hidden_state = mt_model.encoder(src_tensor)
    next_token = torch.zeros(1, 1, dtype = torch.long)
    predicted_words = []

    with torch.no_grad():
        decoder_hidden_state = encoder_last_hidden_state
        decoder_cell_state = torch.zeros(1, 1, decoder_hidden_state.shape[2])
        for i in range(40):
            distributional_prob, decoder_hidden_state, decoder_cell_state = mt_model.decoder.one_step_decoder(next_token, decoder_hidden_state, encoder_hidden_states, decoder_cell_state, attention_mask)
            next_token = distributional_prob.argmax(1).unsqueeze(1)

            predicted = distributional_prob.argmax(1).item()
            if predicted == 1: #EOS
                break
            else:
                predicted_words.append(predicted)
    bleu_scores.append(bleu_score.sentence_bleu(target, predicted_words, smoothing_function=bleu_score.SmoothingFunction().method1))

print("評価データ全体のBLEUスコア（平均）")
print(sum(bleu_scores)/len(bleu_scores)) #0.06554595627012831
