import sentencepiece as spm
import pickle
"""
spm.SentencePieceTrainer.train(input='../kftt-data-1.0/data/orig/kyoto-train.ja', model_prefix='ja_vocab', vocab_size = 8000, bos_id = 0, eos_id = 1, unk_id = 2, pad_id = 7999)
spm.SentencePieceTrainer.train(input='../kftt-data-1.0/data/orig/kyoto-train.en', model_prefix='en_vocab', vocab_size = 8000, bos_id = 0, eos_id = 1, unk_id = 2, pad_id = 7999)
"""
ja_model = spm.SentencePieceProcessor(model_file='ja_vocab.model')
en_model = spm.SentencePieceProcessor(model_file='en_vocab.model')

with open("../kftt-data-1.0/data/orig/kyoto-train.ja", "r") as fr_1:
    ja_sentences = fr_1.readlines()
    ja_sentences = [sentence[:len(sentence)-1] for sentence in ja_sentences]
    with open("../kftt-data-1.0/data/orig/kyoto-train.en", "r") as fr_2:
        en_sentences = fr_2.readlines()
        en_sentences = [sentence[:len(sentence)-1] for sentence in en_sentences]

        encoded_ja_sentences = ja_model.encode(ja_sentences)
        encoded_en_sentences = en_model.encode(en_sentences)

        too_long_ids = []
        for i in range(len(encoded_ja_sentences)):
            if max(len(encoded_ja_sentences[i]), len(encoded_en_sentences[i])) > 50:
                too_long_ids.append(i)

        short_encoded_ja_sentences = []
        short_encoded_en_sentences = []
        for i in range(len(encoded_ja_sentences)):
            if i not in too_long_ids:
                short_encoded_ja_sentences.append(encoded_ja_sentences[i])
                short_encoded_en_sentences.append(encoded_en_sentences[i])

        with open("encoded_ja_train.txt", "wb") as fw_1:
            pickle.dump(short_encoded_ja_sentences, fw_1)

        with open("encoded_en_train.txt", "wb") as fw_2:
            pickle.dump(short_encoded_en_sentences, fw_2)
