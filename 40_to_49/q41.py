class Morph:
    def __init__(self, surface, base, pos, pos1):
        self.surface = surface
        self.base = base
        self.pos = pos
        self.pos1 = pos1

    def morph_print(self):
        print(self.surface, self.base, self.pos, self.pos1)

class Chunk:
    def __init__(self, morphs, dst, srcs):
        self.morphs = morphs
        self.dst = dst
        self.srcs = srcs

    def chunk_print(self):
        print("dst:",self.dst, "srcs:", self.srcs)
        chunk_str = ""
        for i in range(len(self.morphs)):
            chunk_str += self.morphs[i].surface
        print(chunk_str)

chunk_list = []

with open("ai.ja.txt.parsed") as file_1:
    Lines = file_1.readlines()
    eos_counter = 0
    row_num = 0
    chunk_num = -1

    while eos_counter < 2:
        if Lines[row_num] == "EOS\n":
            eos_counter += 1
        row_num += 1
    while eos_counter < 3:
        if Lines[row_num] == "EOS\n":
            eos_counter += 1
        else:
            if Lines[row_num][0] == "*":
                chunk_num += 1
                dependency = Lines[row_num].split(" ")
                temp = Chunk([], int(dependency[2][:len(dependency[2])-1]), [])
                chunk_list.append(temp)
            else:
                cur = Lines[row_num].split("\t")
                other_elements = cur[1].split(",")
                temp = Morph(cur[0], other_elements[6], other_elements[0], other_elements[1])
                chunk_list[chunk_num].morphs.append(temp)
        row_num += 1

    for i in range(len(chunk_list)):
        dst_i = chunk_list[i].dst
        chunk_list[dst_i].srcs.append(i)

for i in range(len(chunk_list)):
    print("index:", i)
    chunk_list[i].chunk_print()
    print()
