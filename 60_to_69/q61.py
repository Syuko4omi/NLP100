from gensim.models import KeyedVectors
import numpy as np

word_vectors_from_bin = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
wv_1 = word_vectors_from_bin.wv["United_States"]
wv_2 = word_vectors_from_bin.wv["U.S."]
print(np.dot(wv_1, wv_2)/(np.linalg.norm(wv_1)*np.linalg.norm(wv_2)))
#0.7310775
