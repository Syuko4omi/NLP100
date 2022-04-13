sent = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."
word_list = []
flag = True
pos = 0
ign = [" ", ".", ","]
while flag:
    temp = ""
    current_char = ""
    while sent[pos] not in ign:
        temp += sent[pos]
        pos += 1
    if len(temp) != 0:
        word_list.append(len(temp))
    if sent[pos] == ".":
        flag = False
    pos += 1
print(word_list)
