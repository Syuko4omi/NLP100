with open("popular-names.txt") as file_1:
    Lines = file_1.readlines()
    name_list = []
    for i in range(len(Lines)):
        temp = Lines[i].split()
        name_list.append(temp[0])

    name_list = list(set(name_list))
    name_list.sort()

    for i in range(len(name_list)):
        print(name_list[i])
