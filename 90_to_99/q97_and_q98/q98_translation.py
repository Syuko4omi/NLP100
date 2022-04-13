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
checkpoint_file = '../trained_model/q98_model/5_10_trained_model_e15.pth'
#checkpoint_file = '../trained_model/q98_model/fine_tuned_trained_model_e15.pth'
checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
mt_model.load_state_dict(checkpoint["model_state_dict"])
ja_model = spm.SentencePieceProcessor(model_file='../q95/ja_vocab.model')
en_model = spm.SentencePieceProcessor(model_file='../q95/en_vocab.model')
mt_model.eval()

while True:
    print("英語に翻訳したい日本語の文章を入力してください")
    src = ja_model.encode([input()])
    src[0].append(1)
    src_tensor = torch.tensor(src)
    attention_mask = torch.ones(1, src_tensor.shape[0], 1)

    encoder_hidden_states, encoder_last_hidden_state = mt_model.encoder(src_tensor)
    next_token = torch.zeros(1, 1, dtype = torch.long)
    predicted_words = []

    with torch.no_grad():
        decoder_hidden_state = encoder_last_hidden_state
        decoder_cell_state = torch.zeros(1, 1, decoder_hidden_state.shape[2])
        for i in range(40):
            distributional_prob, decoder_hidden_state, decoder_cell_state = mt_model.decoder.one_step_decoder(next_token, decoder_hidden_state, encoder_hidden_states, decoder_cell_state, attention_mask)
            next_token = distributional_prob.argmax(1).unsqueeze(0)
            predicted = en_model.decode([distributional_prob.argmax(1).item()])
            if distributional_prob.argmax(1).item() == 1:
                break
            else:
                predicted_words.append(predicted)
    print(" ".join(predicted_words))
    print()
