import pickle

word_freq = {}
word_dict = {}

with open("train.txt", "r") as f_r:
    L = f_r.readlines()
    for i in range(len(L)):
        title = L[i][2:len(L)-1].split()
        for word in title:
            if word not in word_freq:
                word_freq[word] = 1
            else:
                word_freq[word] += 1
    word_ids = sorted(word_freq.items(), key = lambda kv: kv[1], reverse = True)
    next_id = 1
    valid_vocab_num = 0
    for word, num in word_ids:
        if num >= 2:
            word_dict[word] = next_id
            next_id += 1
            valid_vocab_num += 1
        else:
            word_dict[word] = valid_vocab_num+1

    with open("re_word_dict.pkl", "wb") as tf:
        pickle.dump(word_dict, tf)

def return_sequential_id(word_list): #単語列は単語のリストで渡されるとする
    id_list = []
    for word in word_list:
        if word in word_dict:
            id_list.append(word_dict[word])
        else:
            id_list.append(0)
    return id_list

#print(return_sequential_id(["Pound", "Strengthens", "Fifth", "Day", "Versus", "Euro", "on", "BOE", "Rate", "Speculation"]))
