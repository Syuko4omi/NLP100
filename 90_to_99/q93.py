import torch
import torch.nn as nn
import numpy as np
import nltk
from nltk import bleu_score
import pickle

from class_lang import Lang
from encoder_decoder_model import LSTMEncoder, Attention, OneStepDecoder, LSTMDecoderwithAttention, EncoderDecoder

#references = [["I", "have", "a", "pen", "and", "apple"]]
#hypothesis = ["I", "have", "a", "pinapple"]
#bleuscore = bleu_score.sentence_bleu(references, hypothesis, smoothing_function=bleu_score.SmoothingFunction().method1)
#bleuscore = bleu_score.sentence_bleu(references, hypothesis, smoothing_function=bleu_score.SmoothingFunction().method1, weights = (1/2, 1/2))

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

with open("q91/input_lang.pkl", "rb") as fr_1:
    input_lang = pickle.load(fr_1)
with open("q91/output_lang.pkl", "rb") as fr_2:
    output_lang = pickle.load(fr_2)

mt_model = create_model_for_inference(input_lang.num_of_words+3, output_lang.num_of_words+3)
checkpoint_file = 'trained_model/q91_model/trained_model_e9.pth'
checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
mt_model.load_state_dict(checkpoint["model_state_dict"])

def read_data(src_lang, dst_lang, data_type): #ここではまだlangの中身を作らない
    with open(f"kftt-data-1.0/data/tok/kyoto-{data_type}.{src_lang}", "r") as fr_1:
        lines_1 = fr_1.readlines()
        with open(f"kftt-data-1.0/data/tok/kyoto-{data_type}.{dst_lang}", "r") as fr_2:
            lines_2 = fr_2.readlines()
            pairs_of_src_and_dst = [[lines_1[i][:-1].split(), lines_2[i][:-1].split()] for i in range(len(lines_1))]

            return pairs_of_src_and_dst

test_pairs = read_data("ja", "en", "test")
bleu_scores = []

for pair in test_pairs[1:]: #最初のデータは除く
    source = pair[0]
    target = [pair[1]]
    source = [input_lang.word2id[word] for word in source if word in input_lang.word2id]
    source.append(1)
    source_tensor = torch.tensor([source])
    attention_mask = torch.ones(1, source_tensor.shape[0], 1)

    encoder_hidden_states, encoder_last_hidden_state = mt_model.encoder(source_tensor)
    next_token = torch.zeros(1, 1, dtype = torch.long)
    predicted_words = []

    with torch.no_grad():
        decoder_hidden_state = encoder_last_hidden_state
        decoder_cell_state = torch.zeros(1, 1, decoder_hidden_state.shape[2])
        for i in range(40):
            distributional_prob, decoder_hidden_state, decoder_cell_state = mt_model.decoder.one_step_decoder(next_token, decoder_hidden_state, encoder_hidden_states, decoder_cell_state, attention_mask)
            next_token = distributional_prob.argmax(1).unsqueeze(0)

            if distributional_prob.argmax(1).item() != 48405:
                predicted = output_lang.id2word[distributional_prob.argmax(1).item()]
                if predicted == "EOS":
                    break
                else:
                    predicted_words.append(predicted)
            else:
                predicted_words.append("$OOV$")
    #4-gramだと2単語マッチのBLEUが0.3とかで出たりするので参った　どうしてくれる
    #bleu_scores.append(bleu_score.sentence_bleu(target, predicted_words, smoothing_function=bleu_score.SmoothingFunction().method1, weights = (1/2, 1/2)))
    current_bleu_score = bleu_score.sentence_bleu(target, predicted_words, smoothing_function=bleu_score.SmoothingFunction().method1)
    bleu_scores.append(current_bleu_score)
    if current_bleu_score > 0.3:
        print("入力文:", "".join(pair[0]))
        print("予測:", " ".join(predicted_words))
        print("正解:", " ".join(target[0]))
        print("BLEUスコア:", current_bleu_score)
        print()
print("評価データ全体のBLEUスコア（平均）")
print(sum(bleu_scores)/len(bleu_scores)) #0.08673822034190538
