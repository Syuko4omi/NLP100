from gensim.models import KeyedVectors
import numpy as np

word_vectors_from_bin = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
print(word_vectors_from_bin.wv["United_States"])
