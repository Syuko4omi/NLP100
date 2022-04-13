from gensim.models import KeyedVectors
import numpy as np
word_vectors_from_bin = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
#A = word_vectors_from_bin.vocab.keys()

print(word_vectors_from_bin.most_similar(positive=["Spain", "Athens"], negative = ["Madrid"], topn=10))
