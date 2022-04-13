with open("popular-names.txt") as file_1:
    Lines = file_1.readlines()
    temp = []
    for i in range(len(Lines)):
        temp.append(Lines[i].split())
    temp.sort(key = lambda x: int(x[2]), reverse = True)

    for i in range(len(temp)):
        print('\t'.join(temp[i]))
