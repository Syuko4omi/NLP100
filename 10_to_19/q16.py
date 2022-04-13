n = int(input("file_num: "))
suffix = input("suffix: ")

with open("popular-names.txt") as file_1:
    Lines = file_1.readlines()
    line_num = len(Lines)
    rem = line_num%n
    current_line = 0

    for i in range(n):
        file_name = suffix+str(i)
        if rem > 0:
            with open(file_name, "w") as file:
                for i in range((line_num-rem)//n + 1):
                    file.write(Lines[current_line])
                    current_line += 1
                rem -= 1
        else:
            with open(file_name, "w") as file:
                for i in range((line_num-rem)//n):
                    file.write(Lines[current_line])
                    current_line += 1
