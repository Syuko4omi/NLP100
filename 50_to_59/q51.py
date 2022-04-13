from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from nltk import stem

stemmer = stem.PorterStemmer()

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
    #a = vectorizer.get_feature_names()
    #print(len(a))
    print("idf calculated!")

    tf_idf_matrix = vectorizer.transform(train_data_text)
    array_matrix = tf_idf_matrix.toarray()
    np.savetxt("train.feature.txt", array_matrix)
    print("train data done")

    with open("valid.txt", "r") as fr_2:
        raw_valid_data = fr_2.readlines()
        valid_data_text = []
        for i in range(len(raw_valid_data)):
            text_list = raw_valid_data[i][2:len(raw_valid_data[i])-1].split()
            for j in range(len(text_list)):
                text_list[j] = stemmer.stem(text_list[j])
            valid_data_text.append(" ".join(text_list))
        tf_idf_matrix = vectorizer.transform(valid_data_text)
        array_matrix = tf_idf_matrix.toarray()
        np.savetxt("valid.feature.txt", array_matrix)
    print("valid data done")

    with open("test.txt", "r") as fr_3:
        raw_test_data = fr_3.readlines()
        test_data_text = []
        for i in range(len(raw_test_data)):
            text_list = raw_test_data[i][2:len(raw_test_data[i])-1].split()
            for j in range(len(text_list)):
                text_list[j] = stemmer.stem(text_list[j])
            test_data_text.append(" ".join(text_list))
        tf_idf_matrix = vectorizer.transform(test_data_text)
        array_matrix = tf_idf_matrix.toarray()
        np.savetxt("test.feature.txt", array_matrix)
    print("test data done")
