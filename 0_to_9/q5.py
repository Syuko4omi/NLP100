def make_n_gram(seq, n):
    ans_list = []
    for i in range(len(seq)-n+1):
        ans_list.append(seq[i:i+n])
    return ans_list

sent = "I am an NLPer"
for i in range(1, 4):
    print(make_n_gram(sent, i))

word_list = sent.split(" ")
for i in range(1, 4):
    print(make_n_gram(word_list, i))
