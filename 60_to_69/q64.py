import numpy as np
from gensim.models import KeyedVectors

word_vectors_from_bin = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)


with open("questions-words.txt", "r") as f:
    L = f.readlines()
    for i in range(len(L)):
        raw_line = L[i]
        words_list = L[i].split(" ")
        if len(words_list) < 4:
            L[i] = raw_line
            continue
        words_list[3] = words_list[3][:len(words_list[3])-1]
        ans = word_vectors_from_bin.most_similar(positive = [words_list[1], words_list[2]], negative = [words_list[0]])[0]
        L[i] = words_list[0]+" "+words_list[1]+" "+words_list[2]+" "+words_list[3]+" "+ans[0]+" "+str(ans[1])+"\n"
        print(L[i])
    with open("new_questions-words.txt", "w") as fw:
        fw.writelines(L)
