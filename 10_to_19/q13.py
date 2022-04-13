file_1 = open("col1.txt")
file_2 = open("col2.txt")

Lines_1 = file_1.readlines()
Lines_2 = file_2.readlines()

with open("ans_of_q13.txt", "w") as file_3:
    for i in range(len(Lines_1)):
        file_3.write(Lines_1[i][:len(Lines_1[i])-1]+'\t'+Lines_2[i])

file_1.close()
file_2.close()
