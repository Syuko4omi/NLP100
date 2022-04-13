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

A_no_B_list = []

for i in range(len(List_of_sentences)):
    for j in range(len(List_of_sentences[i])):
        if j+2 < len(List_of_sentences[i]):
            if List_of_sentences[i][j]["pos"] == "名詞" and List_of_sentences[i][j+2]["pos"] == "名詞":
                if List_of_sentences[i][j+1]["surface"] == "の":
                    A_no_B_list.append(List_of_sentences[i][j]["surface"]+List_of_sentences[i][j+1]["surface"]+List_of_sentences[i][j+2]["surface"])

#print(A_no_B_list[:10])
