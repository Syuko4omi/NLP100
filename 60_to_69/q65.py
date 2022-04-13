from gensim.models import KeyedVectors
import numpy as np

with open("new_questions-words.txt", "r") as f:
    L = f.readlines()
    sintactic_correct_examples = 0
    sintactic_example_num = 0
    semantic_correct_examples = 0
    semantic_example_num = 0

    mode = "semantic"
    for i in range(len(L)):
        if L[i][:6] == ": gram":
            mode = "sintactic"
        elif L[i][:2] == ": ":
            mode = "semantic"
        else:
            L[i] = L[i].split()
            if mode == "sintactic":
                sintactic_example_num += 1
                if L[i][3] == L[i][4]:
                    sintactic_correct_examples += 1
            else:
                semantic_example_num += 1
                if L[i][3] == L[i][4]:
                    semantic_correct_examples += 1
    print("sintactic_accuracy:", sintactic_correct_examples/sintactic_example_num) #0.7399531615925059
    print("semantic_accuracy:", semantic_correct_examples/semantic_example_num) #0.7308602999210734
