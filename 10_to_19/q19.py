with open("popular-names.txt") as file_1:
    Lines = file_1.readlines()
    temp = []
    for i in range(len(Lines)):
        each_line = Lines[i].split()
        temp.append(each_line[0])

    name_list = list(set(temp))
    counter = [[0, name_list[i]] for i in range(len(name_list))]
    for i in range(len(name_list)):
        for j in range(len(temp)):
            if temp[j] == name_list[i]:
                counter[i][0] += 1

    counter.sort(reverse = True)
    for i in range(len(name_list)):
        print(counter[i][1])
