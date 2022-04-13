sent = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."

#split into some words
ign = [" ", "."]
word_list = []
pos = 0
flag = True
while flag:
    temp = ""
    while sent[pos] not in ign:
        temp += sent[pos]
        pos += 1
    if len(temp) != 0:
        word_list.append(temp)
    if pos == len(sent)-1:
        flag = False
    pos += 1

#making a dictionary
excep = [1,5,6,7,8,9,15,16,19]
my_dict = {}
for i in range(len(word_list)):
    if i+1 in excep:
        sign = word_list[i][0]
    else:
        sign = word_list[i][:2]
    my_dict[sign] = i+1
print(my_dict)
