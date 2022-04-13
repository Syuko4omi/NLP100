from gensim.models import KeyedVectors
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import numpy as np
from matplotlib import pyplot as plt

word_vectors_from_bin = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

with open("countries.txt", "r") as f:
    L = f.readlines()
    counter = 0
    valid_country_name = []
    valid_country_vector = []
    for i in range(len(L)):
        if L[i][:len(L[i])-1] in word_vectors_from_bin:
            valid_country_name.append(L[i][:len(L[i])-1])
    for i in range(len(valid_country_name)):
        valid_country_vector.append(word_vectors_from_bin.wv[valid_country_name[i]])
    z = linkage(valid_country_vector, metric = "euclidean", method = "ward")
    dendrogram(z, labels = valid_country_name)
    plt.show()
