import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pickle

SOS_token = 0
EOS_token = 1

ja_words = open("ja_words.txt", "rb")
en_words = open("en_words.txt", "rb")
ja_words_list = pickle.load(ja_words)
en_words_list = pickle.load(en_words)

class Lang:
    def __init__(self, language):
        self.language = language
        self.word2id = {}
        self.id2word = {0: "SOS", 1: "EOS"}
        self.word2count = {}
        self.num_of_words = 2

    def add_word_to_dict(self, word):
        if self.language == "ja":
            if word not in self.word2id:
                if word in ja_words_list:
                    self.word2id[word] = self.num_of_words
                    self.id2word[self.num_of_words] = word
                    self.word2count[word] = 1
                    self.num_of_words += 1
            else:
                self.word2count[word] += 1
        else:
            if word not in self.word2id:
                if word in en_words_list:
                    self.word2id[word] = self.num_of_words
                    self.id2word[self.num_of_words] = word
                    self.word2count[word] = 1
                    self.num_of_words += 1
            else:
                self.word2count[word] += 1

    def add_sentence_to_dict(self, sentence):
        for word in sentence.split():
            self.add_word_to_dict(word)

def read_data(src_lang, dst_lang, data_type): #ここではまだlangの中身を作らない
    with open(f"kftt-data-1.0/data/tok/kyoto-{data_type}.{src_lang}", "r") as fr_1:
        lines_1 = fr_1.readlines()
        with open(f"kftt-data-1.0/data/tok/kyoto-{data_type}.{dst_lang}", "r") as fr_2:
            lines_2 = fr_2.readlines()
            pairs_of_src_and_dst = [[lines_1[i][:-1], lines_2[i][:-1]] for i in range(len(lines_1))]

            return Lang(src_lang), Lang(dst_lang), pairs_of_src_and_dst

def create_lang(src_lang, dst_lang, data_type): #langの中身を作る(trainのみ使用。testやdevでは使わない)
    input_lang, output_lang, pairs = read_data(src_lang, dst_lang, data_type)
    for pair in pairs:
        input_lang.add_sentence_to_dict(pair[0])
        output_lang.add_sentence_to_dict(pair[1])
    print("Counted words:")
    print(input_lang.language, input_lang.num_of_words)
    print(output_lang.language, output_lang.num_of_words)
    return input_lang, output_lang, pairs

input_lang, output_lang, pairs = create_lang("ja", "en", "train.cln")
with open("input_lang.pkl", "wb") as fw_1:
    pickle.dump(input_lang, fw_1)
with open("output_lang.pkl", "wb") as fw_2:
    pickle.dump(output_lang, fw_2)
with open("train_pairs.txt", "wb") as fw_3:
    pickle.dump(pairs, fw_3)
