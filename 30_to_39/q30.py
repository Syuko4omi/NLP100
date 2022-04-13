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

#print(List_of_sentences[:10])
