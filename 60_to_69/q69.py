from gensim.models import KeyedVectors
from sklearn import datasets
from sklearn.manifold import TSNE
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
    reduced_vectors = TSNE().fit_transform(np.array(valid_country_vector))
    plt.scatter([reduced_vectors[i][0] for i in range(len(valid_country_name))], [reduced_vectors[i][1] for i in range(len(valid_country_name))])
    for i in range(len(valid_country_name)):
        plt.annotate(valid_country_name[i], (reduced_vectors[i][0], reduced_vectors[i][1]))
    plt.show()
