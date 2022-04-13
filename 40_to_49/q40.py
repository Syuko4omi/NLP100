class Morph:
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1

    def my_print(self):
        print(self.surface, self.base, self.pos, self.pos1)

sentence = []

with open("ai.ja.txt.parsed") as file_1:
    Lines = file_1.readlines()
    eos_counter = 0
    row_num = 0
    while eos_counter < 2:
        if Lines[row_num] == "EOS\n":
            eos_counter += 1
        row_num += 1
    while eos_counter < 3:
        if Lines[row_num] == "EOS\n":
            eos_counter += 1
        else:
            if Lines[row_num][0] == "*":
                None
            else:
                cur = Lines[row_num].split("\t")
                other_elements = cur[1].split(",")
                temp = Morph(cur[0], other_elements[6], other_elements[0], other_elements[1])
                sentence.append(temp)
        row_num += 1

for i in range(len(sentence)):
    sentence[i].my_print()
