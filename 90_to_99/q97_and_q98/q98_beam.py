import torch
import torch.nn as nn
import numpy as np
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

mt_model = create_model_for_inference(8000, 8000)
#checkpoint_file = '../trained_model/q98_model/5_10_trained_model_e15.pth'
checkpoint_file = '../trained_model/q98_model/5_10_fine_tuned_trained_model_e15.pth'
checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
mt_model.load_state_dict(checkpoint["model_state_dict"])
ja_model = spm.SentencePieceProcessor(model_file='../q95/ja_vocab.model')
en_model = spm.SentencePieceProcessor(model_file='../q95/en_vocab.model')

BEAM_SIZE = 10

def beam_search(model, src, beam_size):
    src[0].append(1)
    src_tensor = torch.tensor(src)
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
                    if next_tokens[k] != 1: #EOS
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

while True:
    print("英語に翻訳したい日本語の文章を入力してください")
    src = ja_model.encode([input()])
    confirmed_seqs = beam_search(mt_model, src, BEAM_SIZE)

    for i in range(len(confirmed_seqs)):
        temp_seq = confirmed_seqs[i][0]
        temp_prob = confirmed_seqs[i][1]
        output_seq = []
        for j in range(len(temp_seq)):
            output_seq.append(en_model.decode([temp_seq[j]]))
        print(" ".join(output_seq))
        print(f"prob: {temp_prob}")
