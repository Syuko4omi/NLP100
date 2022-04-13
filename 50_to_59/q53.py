from sklearn.linear_model import LogisticRegression
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import stem

stemmer = stem.PorterStemmer()
logreg_model = pickle.load(open("preserved_model.sav", 'rb'))
label_name_list = logreg_model.classes_

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
    print("press title")
    input_title = input().split()
    for j in range(len(input_title)):
        input_title[j] = stemmer.stem(input_title[j])
    X = vectorizer.transform([" ".join(input_title)])
    prob = logreg_model.predict_proba(X)
    for i in range(4):
        print(f"prob of label {label_name_list[i]}: {prob[0][i]}")
