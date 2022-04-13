from gensim.models import KeyedVectors
import numpy as np

word_vectors_from_bin = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

def find_grv(vec_list):
    vec = np.zeros(np.shape(vec_list[0][1])) #なんかそのままだとimmutableなので初期化
    for i in range(1, len(vec_list)):
        vec += vec_list[i][1]
    vec = vec/len(vec_list)
    return vec

def calc_dist(vec_1, vec_2):
    dist = np.sqrt(np.sum(np.square(vec_1-vec_2)))
    return dist

def closest_grv(vec, grv_list):
    temp_dist = float("inf")
    id = -1
    for i in range(len(grv_list)):
        dist = calc_dist(vec, grv_list[i])
        if temp_dist > dist:
            temp_dist = dist
            id = i
    return id

with open("countries.txt", "r") as f:
    L = f.readlines()
    counter = 0
    valid_country_name = []
    for i in range(len(L)):
        if L[i][:len(L[i])-1] in word_vectors_from_bin:
            valid_country_name.append(L[i][:len(L[i])-1])
    initial_cluster = [np.random.randint(0, 5) for i in range(len(valid_country_name))]
    prev_cluster_list = [[] for i in range(5)]
    cur_cluster_list = [[] for i in range(5)]
    for i in range(len(valid_country_name)):
        cur_cluster_list[initial_cluster[i]].append([valid_country_name[i], word_vectors_from_bin.wv[valid_country_name[i]]])
    while prev_cluster_list != cur_cluster_list:
        print([len(cur_cluster_list[i]) for i in range(5)])
        prev_cluster_list = cur_cluster_list
        cur_cluster_list = [[] for i in range(5)]
        grv_list = [find_grv(prev_cluster_list[i]) for i in range(5)]
        for i in range(5):
            for j in range(len(prev_cluster_list[i])):
                closest_grv_id = closest_grv(prev_cluster_list[i][j][1], grv_list)
                cur_cluster_list[closest_grv_id].append(prev_cluster_list[i][j])
    for i in range(5):
        print([cur_cluster_list[i][j][0] for j in range(len(cur_cluster_list[i]))])
