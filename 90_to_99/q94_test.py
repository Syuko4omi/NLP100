import torch
import torch.nn as nn
import numpy as np
import MeCab
import pickle
import nltk
from nltk import bleu_score

from class_lang import Lang
from encoder_decoder_model import LSTMEncoder, Attention, OneStepDecoder, LSTMDecoderwithAttention, EncoderDecoder
from matplotlib import pyplot as plt

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

def read_data(src_lang, dst_lang, data_type): #ここではまだlangの中身を作らない
    with open(f"kftt-data-1.0/data/tok/kyoto-{data_type}.{src_lang}", "r") as fr_1:
        lines_1 = fr_1.readlines()
        with open(f"kftt-data-1.0/data/tok/kyoto-{data_type}.{dst_lang}", "r") as fr_2:
            lines_2 = fr_2.readlines()
            pairs_of_src_and_dst = [[lines_1[i][:-1].split(), lines_2[i][:-1].split()] for i in range(len(lines_1))]

            return pairs_of_src_and_dst

test_pairs = read_data("ja", "en", "test")

mt_model = create_model_for_inference(input_lang.num_of_words+3, output_lang.num_of_words+3)
checkpoint_file = 'trained_model/q91_model/trained_model_e9.pth'
checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
mt_model.load_state_dict(checkpoint["model_state_dict"])
mt_model.eval()

wakati = MeCab.Tagger("-Owakati")

def beam_search(model, src, beam_size):
    src = [input_lang.word2id[word] for word in src if word in input_lang.word2id]
    src.append(1)
    src_tensor = torch.tensor(src).unsqueeze(0)
    attention_mask = torch.ones(1, src_tensor.shape[0], 1)

    encoder_hidden_states, encoder_last_hidden_state = mt_model.encoder(src_tensor)
    next_token = torch.zeros(1, 1, dtype = torch.long)
    predicted_words = []
    confirmed_seqs = []

    with torch.no_grad():
        decoder_hidden_state = encoder_last_hidden_state
        decoder_cell_state = torch.zeros(1, 1, decoder_hidden_state.shape[2])
        remained_seqs = [[[], torch.tensor([[0]]), 1.0, decoder_hidden_state, decoder_cell_state]]
        for i in range(40):
            bfs_buffer = []
            for j in range(len(remained_seqs)):
                temporally_predicted_tokens, last_prediction, prob, decoder_hidden_state, decoder_cell_state = remained_seqs[j]
                distributional_prob, decoder_hidden_state, decoder_cell_state = mt_model.decoder.one_step_decoder(last_prediction, decoder_hidden_state, encoder_hidden_states, decoder_cell_state, attention_mask)
                next_tokens = np.argpartition(-distributional_prob, beam_size)[0][:beam_size]
                probs = np.array([distributional_prob[0][id] for id in next_tokens])*prob
                for k in range(len(next_tokens)):
                    if next_tokens[k] != 1:
                        bfs_buffer.append([temporally_predicted_tokens+[next_tokens[k].item()], next_tokens[k].unsqueeze(0).unsqueeze(0), probs[k], decoder_hidden_state, decoder_cell_state])
                    else:
                        confirmed_seqs.append([temporally_predicted_tokens, probs[k]])
            bfs_buffer.sort(key = lambda x: x[2], reverse = True)
            remained_seqs = bfs_buffer[:beam_size]

        for i in range(len(remained_seqs)):
            confirmed_seqs.append([remained_seqs[i][0], remained_seqs[i][2]])
        confirmed_seqs.sort(key = lambda x: x[1], reverse = True)
        confirmed_seqs = confirmed_seqs[:min(beam_size, len(confirmed_seqs))]

        return confirmed_seqs


def calc_bleu_according_to_beam_size(beam_size):
    bleu_scores = []
    counter = 0
    for pair in test_pairs[1:]: #最初のデータは除く
        source = pair[0]
        target = [pair[1]]
        confirmed_seq = beam_search(mt_model, source, beam_size)[0][0]
        output_seq = []
        for j in range(len(confirmed_seq)):
            if confirmed_seq[j] in output_lang.id2word:
                output_seq.append(output_lang.id2word[confirmed_seq[j]])
            else:
                output_seq.append("$OOV$")
        current_bleu_score = bleu_score.sentence_bleu(target, output_seq, smoothing_function=bleu_score.SmoothingFunction().method1)
        bleu_scores.append(current_bleu_score)
        counter += 1
        if counter%100 == 0:
            print(f"{counter} / {len(test_pairs[1:])} done")
            break #全部見てるとメチャクチャ時間かかるので100個だけに変更
    return sum(bleu_scores)/len(bleu_scores)

L = []
beam_size_cand = [1, 2, 5, 10]
blue_score_for_each_beam_size = []
for i in range(len(beam_size_cand)):
    temp = calc_bleu_according_to_beam_size(beam_size_cand[i])
    blue_score_for_each_beam_size.append(temp)

plt.plot(np.array(beam_size_cand), np.array(blue_score_for_each_beam_size))
plt.show()
