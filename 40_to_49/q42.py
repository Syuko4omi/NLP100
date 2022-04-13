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
        if Lines[i] == "EOS\n": #一文まるっと読んだら、その文に含まれる文節のリストを出力して空にする。
            for j in range(len(chunk_list)):
                dst_j = chunk_list[j].dst
                if dst_j != -1:
                    chunk_list[dst_j].srcs.append(j)
            for j in range(len(chunk_list)):
                dest_chunk = chunk_list[j].make_chunk()
                for k in range(len(chunk_list[j].srcs)):
                    src_chunk = chunk_list[chunk_list[j].srcs[k]].make_chunk()
                    print(src_chunk+"\t"+dest_chunk)
            chunk_list = []
            chunk_num = -1
            print()
        elif Lines[i][0] == "*":
            chunk_num += 1
            dependency = Lines[i].split(" ")
            temp = Chunk([], int(dependency[2][:len(dependency[2])-1]), []) #文節が含む形態素のリストと係り元は空にしておいて、else以下でいれていく
            chunk_list.append(temp)
        else:
            cur = Lines[i].split("\t")
            other_elements = cur[1].split(",")
            temp = Morph(cur[0], other_elements[6], other_elements[0], other_elements[1])
            chunk_list[chunk_num].morphs.append(temp)
