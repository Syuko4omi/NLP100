lines = []
new_lines = []

with open("popular-names.txt") as file_1:
    lines = file_1.readlines()
    for i in range(len(lines)):
        temp = ""
        for j in range(len(lines[i])):
            if lines[i][j] == '\t':
                temp += ' '
            else:
                temp += lines[i][j]
        new_lines.append(temp)

with open("ans_of_q11.txt", 'w') as file_2:
    for i in range(len(new_lines)):
        file_2.write(new_lines[i])
