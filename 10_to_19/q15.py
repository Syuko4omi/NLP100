n = int(input())

with open("popular-names.txt") as file_1:
    Lines = file_1.readlines()
    for i in range(n):
        cur = len(Lines)-n+i
        print(Lines[cur][:len(Lines[cur])-1])
