import sentencepiece as spm
import pickle
ja_model = spm.SentencePieceProcessor(model_file='q95/ja_vocab.model')
en_model = spm.SentencePieceProcessor(model_file='q95/en_vocab.model')

"""
with open("shorter_en-ja.bicleaner05.txt", "r") as fr_1:
    with open("shorter_and_more_precise_en-ja.bicleaner05.txt", "w") as fw_1:
        for line in fr_1:
            L = line.split("\t")
            num = L[0]
            en_sentence = L[1]
            ja_sentence = L[2]
            encoded_en_sentence = en_model.encode(en_sentence)
            encoded_ja_sentence = ja_model.encode(ja_sentence[:len(ja_sentence)-1])
            if float(num) >= 0.785:
                if max(len(encoded_ja_sentence), len(encoded_en_sentence)) <= 50:
                    fw_1.write(num+"\t"+en_sentence+"\t"+ja_sentence)
"""
with open("shorter_and_more_precise_en-ja.bicleaner05.txt", "r") as fr_1:
    lines = fr_1.readlines()
    encoded_en_sentences = []
    encoded_ja_sentences = []
    counter = 0
    for line in lines:
        L = line.split("\t")
        en_sentence = L[1]
        ja_sentence = L[2]
        encoded_en_sentence = en_model.encode(en_sentence)
        encoded_ja_sentence = ja_model.encode(ja_sentence[:len(ja_sentence)-1])
        encoded_en_sentences.append(encoded_en_sentence)
        encoded_ja_sentences.append(encoded_ja_sentence)

    with open("encoded_ja_fine_tune.txt", "wb") as fw_1:
        pickle.dump(encoded_ja_sentences, fw_1)

    with open("encoded_en_fine_tune.txt", "wb") as fw_2:
        pickle.dump(encoded_en_sentences, fw_2)
