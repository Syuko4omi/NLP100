n = int(input())

with open("popular-names.txt") as file_1:
    Lines = file_1.readlines()
    for i in range(n):
        print(Lines[i][:len(Lines[i])-1])
