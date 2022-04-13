import random

sent = "I couldn't believe that I could actually understand what I was reading : the phenomenal power of the human mind ."

def change_char_order(word):
    if len(word) <= 4:
        return word
    else:
        temp = word[0]
        index_list = list(range(1, len(word)-1))
        random.shuffle(index_list)
        for i in range(len(index_list)):
            temp += word[index_list[i]]
        return temp+word[len(word)-1]

word_list = []
pos = 0
temp = ""
ans = ""
for i in range(len(sent)):
    if sent[pos] != " ":
        temp += sent[pos]
        pos += 1
        if pos == len(sent):
            word_list.append(temp)
    else:
        word_list.append(temp)
        pos += 1
        temp = ""
for i in range(len(word_list)):
    word_list[i] = change_char_order(word_list[i])
print(" ".join(word_list))
