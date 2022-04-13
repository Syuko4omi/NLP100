import sentencepiece as spm
import pickle

ja_model = spm.SentencePieceProcessor(model_file='../q95/ja_vocab.model')
en_model = spm.SentencePieceProcessor(model_file='../q95/en_vocab.model')

counter = 0
written_sentence_num = 0
with open("en-ja/en-ja.bicleaner05.txt", "r") as fr_1:
    with open("shorter_en-ja.bicleaner05.txt", "w") as fw_1:
        for line in fr_1:
            L = line.split("\t")
            num = L[2]
            en_sentence = L[3]
            ja_sentence = L[4]
            encoded_en_sentence = en_model.encode(en_sentence)
            encoded_ja_sentence = ja_model.encode(ja_sentence[:len(ja_sentence)-1])
            if float(num) >= 0.775:
                if max(len(encoded_ja_sentence), len(encoded_en_sentence)) <= 50:
                    fw_1.write(num+"\t"+en_sentence+"\t"+ja_sentence)
                    written_sentence_num += 1
            counter += 1
            if counter % 100000 == 0:
                print(f"{counter} sentence processed")
                print(f"wrote {written_sentence_num} sentences")
