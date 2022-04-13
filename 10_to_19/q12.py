col_1 = []
col_2 = []

with open("popular-names.txt") as file_1:
    Lines = file_1.readlines()

for i in range(len(Lines)):
    temp = Lines[i].split()
    col_1.append(temp[0])
    col_2.append(temp[1])

with open("col1.txt", "w") as file_2:
    for i in range(len(col_1)):
        file_2.write(col_1[i]+'\n')

with open("col2.txt", "w") as file_3:
    for i in range(len(col_2)):
        file_3.write(col_2[i]+'\n')
