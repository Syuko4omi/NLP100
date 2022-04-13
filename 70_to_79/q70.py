import numpy as np
import gensim
from gensim.models import KeyedVectors

word_vectors = KeyedVectors.load_word2vec_format("../60_to_69/GoogleNews-vectors-negative300.bin", binary=True)
indexes = {"b":0, "t":1, "e":2, "m":3}

def create_files(data_type):
    with open(f"{data_type}.txt") as f_r:
        L = f_r.readlines()
        Labels = []
        Sentence_Embeddings = []
        for i in range(len(L)):
            tokens = L[i][:len(L[i])-1].split()
            Labels.append(indexes[tokens[0]])
            embedding = np.zeros(300)
            item_num = 0
            for j in range(1, len(tokens)):
                if tokens[j] in word_vectors:
                    temp = word_vectors[tokens[j]]
                    embedding += temp
                    item_num += 1
            if item_num != 0:
                embedding /= item_num
            Sentence_Embeddings.append(embedding)
        np.save(f"{data_type}_feature", np.array(Sentence_Embeddings))
        np.save(f"{data_type}_label", np.array(Labels))

create_files("train")
create_files("valid")
create_files("test")
