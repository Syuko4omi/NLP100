from sklearn.metrics import classification_report
import pickle
import numpy as np
from nltk import stem
from sklearn.feature_extraction.text import TfidfVectorizer

stemmer = stem.PorterStemmer()

logreg_model = pickle.load(open("preserved_model.sav", 'rb'))

coefficient_matrix = logreg_model.coef_
class_name = logreg_model.classes_

with open("train.txt", "r") as fr_1:
    vectorizer = TfidfVectorizer()
    raw_train_data = fr_1.readlines()
    train_data_text = []
    for i in range(len(raw_train_data)):
        text_list = raw_train_data[i][2:len(raw_train_data[i])-1].split()
        for j in range(len(text_list)):
            text_list[j] = stemmer.stem(text_list[j])
        train_data_text.append(" ".join(text_list))
    vectorizer.fit(train_data_text)
    feature_name_list = vectorizer.get_feature_names()
    for i in range(4):
        cur_list = np.argsort(abs(coefficient_matrix[i]))[::-1]
        print(f"category {class_name[i]}")
        print("Top 10")
        for j in range(10):
            print([feature_name_list[cur_list[j]], coefficient_matrix[i][cur_list[j]]])
        print("Worst 10")
        for j in range(len(feature_name_list)-1, len(feature_name_list)-11, -1):
            print([feature_name_list[cur_list[j]], coefficient_matrix[i][cur_list[j]]])

"""
stemmer = stem.PorterStemmer()

logreg_model = pickle.load(open("preserved_model.sav", 'rb'))

coefficient_matrix = logreg_model.coef_
feature_weight = []
word_num = coefficient_matrix[0]
for i in range(len(word_num)):
    weight_i = 0
    for j in range(4):
        weight_i += coefficient_matrix[j][i]**2
    feature_weight.append(weight_i**0.5)

with open("train.txt", "r") as fr_1:
    vectorizer = TfidfVectorizer()
    raw_train_data = fr_1.readlines()
    train_data_text = []
    for i in range(len(raw_train_data)):
        text_list = raw_train_data[i][2:len(raw_train_data[i])-1].split()
        for j in range(len(text_list)):
            text_list[j] = stemmer.stem(text_list[j])
        train_data_text.append(" ".join(text_list))
    vectorizer.fit(train_data_text)
    feature_name_list = vectorizer.get_feature_names()
    L = [[feature_weight[i], feature_name_list[i]] for i in range(len(feature_name_list))]
    L.sort(reverse = True)
    print(L[:10])
    print(L[len(L)-10:])
"""
