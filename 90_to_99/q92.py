import torch
import torch.nn as nn
import numpy as np
import MeCab
import pickle

from class_lang import Lang
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

with open("q91/input_lang.pkl", "rb") as fr_1:
    input_lang = pickle.load(fr_1)
with open("q91/output_lang.pkl", "rb") as fr_2:
    output_lang = pickle.load(fr_2)

mt_model = create_model_for_inference(input_lang.num_of_words+3, output_lang.num_of_words+3)
checkpoint_file = "trained_model/q91_model/trained_model_e9.pth"
checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
mt_model.load_state_dict(checkpoint["model_state_dict"])
wakati = MeCab.Tagger("-Owakati")
mt_model.eval()

while True:
    print("英語に翻訳したい日本語の文章を入力してください")
    src = wakati.parse(input()).rstrip().split()
    src = [input_lang.word2id[word] for word in src if word in input_lang.word2id]
    src.append(1)
    src_tensor = torch.tensor(src).unsqueeze(0)
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
            if distributional_prob.argmax(1).item() in output_lang.id2word:
                predicted = output_lang.id2word[distributional_prob.argmax(1).item()]
                if predicted == "EOS":
                    break
                else:
                    predicted_words.append(predicted)
            else:
                if distributional_prob.argmax(1).item() == 48405: #OOV
                    predicted_words.append("$OOV$")
                else: #PADDING
                    None
    print(" ".join(predicted_words))
    print()
