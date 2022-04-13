from gensim.models import KeyedVectors
import numpy as np

word_vectors_from_bin = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

with open("wordsim353/combined.csv", "r") as f:
    L = f.readlines()
    human_judgement = []
    word2vec_judgement = []
    pair_num = len(L)-1
    for i in range(1, len(L)):
        words_list = L[i].split(",")
        words_list[2] = float(words_list[2][:len(words_list[2])-1])
        words_list.append(i)
        human_judgement.append(words_list)
    human_judgement.sort(key = lambda x: x[2])

    for i in range(pair_num):
        word_1 = human_judgement[i][0]
        word_2 = human_judgement[i][1]
        word_similarity = word_vectors_from_bin.similarity(word_1, word_2)
        id = human_judgement[i][3]
        word2vec_judgement.append([word_1, word_2, word_similarity, id])
    word2vec_judgement.sort(key = lambda x: x[2])

    #print(human_judgement[:5])
    #print(word2vec_judgement[:5])

    squared_rank_diff = []
    for i in range(pair_num):
        current_pair_id = human_judgement[i][3]
        rank = 0
        for j in range(pair_num):
            if word2vec_judgement[j][3] == current_pair_id:
                rank = j
                break
        squared_rank_diff.append((rank-i)**2)

    rho = 1 - ((6*sum(squared_rank_diff))/(pair_num**3 - pair_num))
    print("Spearman's rank correlation coefficient:", rho) #0.7000378114946944
