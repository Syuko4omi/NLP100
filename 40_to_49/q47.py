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
        prohibited_list = ["。", "、"]
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
                leftmost_verb = ""
                for k in range(len(chunk_list[j].morphs)):
                    if chunk_list[j].morphs[k].pos == "動詞":
                        if leftmost_verb == "":
                            leftmost_verb = chunk_list[j].morphs[k].base
                case_and_term_list = []
                disp_checker = False
                sahen_noun = ""
                for k in range(len(chunk_list[j].srcs)):
                    for l in range(len(chunk_list[chunk_list[j].srcs[k]].morphs)):
                        if chunk_list[chunk_list[j].srcs[k]].morphs[l].pos == "助詞":
                            src_chunk = chunk_list[chunk_list[j].srcs[k]].make_chunk()
                            case_and_term_list.append([chunk_list[chunk_list[j].srcs[k]].morphs[l].surface, src_chunk])
                    for l in range(len(chunk_list[chunk_list[j].srcs[k]].morphs)-1):
                        if chunk_list[chunk_list[j].srcs[k]].morphs[l].pos1 == "サ変接続" and chunk_list[chunk_list[j].srcs[k]].morphs[l+1].pos == "助詞" and chunk_list[chunk_list[j].srcs[k]].morphs[l+1].surface == "を":
                            disp_checker = True
                            sahen_noun = chunk_list[chunk_list[j].srcs[k]].morphs[l].surface
                case_and_term_list.sort()
                case_list = [case_and_term_list[i][0] for i in range(len(case_and_term_list))]
                term_list = [case_and_term_list[i][1] for i in range(len(case_and_term_list))]
                if len(case_list) != 0 and leftmost_verb != "" and disp_checker == True:
                    print(sahen_noun+"を"+leftmost_verb+"\t", " ".join(case_list), " ".join(term_list))
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
