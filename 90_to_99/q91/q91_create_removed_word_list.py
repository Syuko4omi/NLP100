import pickle

with open("kftt-data-1.0/data/tok/kyoto-train.cln.ja", "r") as fr_1:
    lines_1 = fr_1.readlines()
    word_list_ja = [lines_1[i][:-1].split() for i in range(len(lines_1))]
    word_dict_ja = {}
    for i in range(len(word_list_ja)):
        for j in range(len(word_list_ja[i])):
            if word_list_ja[i][j] not in word_dict_ja:
                word_dict_ja[word_list_ja[i][j]] = 1
            else:
                word_dict_ja[word_list_ja[i][j]] += 1
                
    with open("kftt-data-1.0/data/tok/kyoto-train.cln.en", "r") as fr_2:
        lines_2 = fr_2.readlines()
        word_list_en = [lines_2[i][:-1].split() for i in range(len(lines_2))]
        word_dict_en = {}
        for i in range(len(word_list_en)):
            for j in range(len(word_list_en[i])):
                if word_list_en[i][j] not in word_dict_en:
                    word_dict_en[word_list_en[i][j]] = 1
                else:
                    word_dict_en[word_list_en[i][j]] += 1

        ja_words_appear_more_than_twice = []
        en_words_appear_more_than_twice = []
        for ja_key in word_dict_ja.keys():
            if word_dict_ja[ja_key] > 3:
                ja_words_appear_more_than_twice.append(ja_key)
        for en_key in word_dict_en.keys():
            if word_dict_en[en_key] > 3:
                en_words_appear_more_than_twice.append(en_key)

        fw_1 = open("ja_words.txt", "wb")
        pickle.dump(ja_words_appear_more_than_twice, fw_1)
        print(f"Japanese words: {len(ja_words_appear_more_than_twice)}")
        fw_2 = open("en_words.txt", "wb")
        pickle.dump(en_words_appear_more_than_twice, fw_2)
        print(f"English words: {len(en_words_appear_more_than_twice)}")
        fw_1.close()
        fw_2.close()
