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

    def make_chunk(self):
        chunk_str = ""
        prohibited_list = ["。", "、", "「", "」", "『", "』", "（", "）"]
        for i in range(len(self.morphs)):
            if self.morphs[i].surface not in prohibited_list:
                chunk_str += self.morphs[i].surface
        return chunk_str

    def chunk_print(self):
        print("dst:",self.dst, "srcs:", self.srcs)
        chunk_str = self.make_chunk()
        print(chunk_str)


with open("ai.ja.txt.parsed") as file_1:
    Lines = file_1.readlines()
    chunk_list = []
    chunk_num = -1

    for i in range(len(Lines)):
        if Lines[i] == "EOS\n":
            for j in range(len(chunk_list)):
                dst_j = chunk_list[j].dst
                if dst_j != -1:
                    chunk_list[dst_j].srcs.append(j)
            for j in range(len(chunk_list)):
                path_elements = []
                noun_flag = False
                for k in range(len(chunk_list[j].morphs)):
                    if chunk_list[j].morphs[k].pos == "名詞":
                        noun_flag = True
                if noun_flag:
                    chunk_including_noun = chunk_list[j].make_chunk()
                    path_elements.append(chunk_including_noun)
                current_id = j
                while chunk_list[current_id].dst != -1:
                    prev_id = current_id
                    current_id = chunk_list[current_id].dst
                    current_chunk = chunk_list[current_id].make_chunk()
                    break_flag = False
                    for k in range(current_id-prev_id):
                        for l in range(len(chunk_list[prev_id+k].morphs)):
                            if chunk_list[prev_id+k].morphs[l].surface == "。":
                                break_flag = True
                    if not break_flag:
                        path_elements.append(current_chunk)
                    else:
                        break
                if len(path_elements) > 1 and noun_flag:
                    print(" -> ".join(path_elements))
            chunk_list = []
            chunk_num = -1
        elif Lines[i][0] == "*":
            chunk_num += 1
            dependency = Lines[i].split(" ")
            temp = Chunk([], int(dependency[2][:len(dependency[2])-1]), [])
            chunk_list.append(temp)
        else:
            cur = Lines[i].split("\t")
            other_elements = cur[1].split(",")
            temp = Morph(cur[0], other_elements[6], other_elements[0], other_elements[1])
            chunk_list[chunk_num].morphs.append(temp)
