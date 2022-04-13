import numpy as np
import matplotlib.pyplot as plt

List_of_sentences = []

with open("neko.txt.mecab") as file_1:
    All_rows = file_1.readlines()
    temp = []

    for i in range(len(All_rows)):
        if All_rows[i] == "EOS\n":
            if len(temp) != 0:
                List_of_sentences.append(temp)
                temp = []
        elif All_rows[i][0] == "\t" or All_rows[i][0] == "\n":
            None
        else:
            surface_ = ""
            pos = 0
            while All_rows[i][pos] != "\t":
                surface_ += All_rows[i][pos]
                pos += 1
            other_elements = All_rows[i][pos+1:len(All_rows[i])-1].split(",")
            base_ = other_elements[6] #基本形（原形）
            pos_ = other_elements[0] #品詞
            pos1_ = other_elements[1] #品詞細分類1
            temp.append({"surface": surface_, "base": base_, "pos": pos_, "pos1": pos1_})

co_occurance_dict = {}
for i in range(len(List_of_sentences)):
    flag = False
    for j in range(len(List_of_sentences[i])):
        if List_of_sentences[i][j]["surface"] == "猫":
            flag = True
    if flag:
        for j in range(len(List_of_sentences[i])):
            temp_word = List_of_sentences[i][j]["surface"]
            if temp_word != "猫":
                if temp_word not in co_occurance_dict:
                    co_occurance_dict[temp_word] = 1
                else:
                    co_occurance_dict[temp_word] += 1

sorted_list = sorted(co_occurance_dict.items(), key = lambda x: x[1], reverse = True)

top_10_word = [sorted_list[i][0] for i in range(10)]
top_10_freq = np.array([sorted_list[i][1] for i in range(10)])
plt.bar(np.array([i for i in range(10)]), top_10_freq, tick_label = top_10_word)
plt.show()
