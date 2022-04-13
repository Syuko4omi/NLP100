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

def make_path_elements_list(chunk_list, id): #名詞を含む文節のインデックスを渡して、そこから根に至るパスを返す
    path_elements = []
    id_list = []
    chunk_including_noun = chunk_list[id].make_chunk()
    path_elements.append([chunk_including_noun, id])
    id_list.append(id)
    current_id = id
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
            path_elements.append([current_chunk, current_id])
            id_list.append(current_id)
        else:
            break
    return path_elements, id_list


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
            id_of_chunk_including_noun = []
            for j in range(len(chunk_list)):
                path_elements = []
                noun_flag = False
                for k in range(len(chunk_list[j].morphs)):
                    if chunk_list[j].morphs[k].pos == "名詞":
                        noun_flag = True
                if noun_flag:
                    id_of_chunk_including_noun.append(j)
            for j in range(len(id_of_chunk_including_noun)):
                j_ = id_of_chunk_including_noun[j]
                path_ele_of_j, id_list_of_j = make_path_elements_list(chunk_list, j_)
                for k in range(j+1, len(id_of_chunk_including_noun)):
                    k_ = id_of_chunk_including_noun[k]
                    path_ele_of_k, id_list_of_k = make_path_elements_list(chunk_list, k_)
                    if k_ in id_list_of_j: #根に向かうパスの途中にjがある
                        cur = id_list_of_j[1]
                        ans_list = ["X"]
                        id_inside_of_path_ele_list = 1
                        check_flag = True
                        for l in range(len(chunk_list[j_].morphs)): #最初の名詞をXに置き換える処理
                            if chunk_list[j_].morphs[l].pos == "名詞" and check_flag:
                                None
                            elif chunk_list[j_].morphs[l].pos == "記号" and check_flag:
                                if l+1 < len(chunk_list[j_].morphs):
                                    if chunk_list[j_].morphs[l+1].pos == "名詞":
                                        None
                            else:
                                check_flag = False
                                ans_list[0] += chunk_list[j_].morphs[l].surface
                        while cur != k_:
                            ans_list.append(path_ele_of_j[id_inside_of_path_ele_list][0])
                            id_inside_of_path_ele_list += 1
                            cur = id_list_of_j[id_inside_of_path_ele_list]
                        ans_list.append("Y")
                        check_flag = True
                        for l in range(len(chunk_list[k_].morphs)):
                            if chunk_list[k_].morphs[l].pos == "名詞" and check_flag:
                                None
                            elif chunk_list[k_].morphs[l].pos == "記号" and check_flag:
                                if l+1 < len(chunk_list[k_].morphs):
                                    if chunk_list[k_].morphs[l+1].pos == "名詞":
                                        None
                            else:
                                check_flag = False
                                ans_list[len(ans_list)-1] += chunk_list[k_].morphs[l].surface
                        print(" -> ".join(ans_list))
                    elif path_ele_of_j[len(path_ele_of_j)-1][0] == path_ele_of_k[len(path_ele_of_k)-1][0]: #共通の根を持ち、i -> jという経路がない
                        X_list = ["X"]
                        Y_list = ["Y"]
                        check_flag = True
                        for l in range(len(chunk_list[j_].morphs)):
                            if chunk_list[j_].morphs[l].pos == "名詞" and check_flag:
                                None
                            elif chunk_list[j_].morphs[l].pos == "記号" and check_flag:
                                if l+1 < len(chunk_list[j_].morphs):
                                    if chunk_list[j_].morphs[l+1].pos == "名詞":
                                        None
                            else:
                                check_flag = False
                                X_list[0] += chunk_list[j_].morphs[l].surface
                        check_flag = True
                        for l in range(len(chunk_list[k_].morphs)):
                            if chunk_list[k_].morphs[l].pos == "名詞" and check_flag:
                                None
                            elif chunk_list[k_].morphs[l].pos == "記号" and check_flag:
                                if l+1 < len(chunk_list[k_].morphs):
                                    if chunk_list[k_].morphs[l+1].pos == "名詞":
                                        None
                            else:
                                check_flag = False
                                Y_list[0] += chunk_list[k_].morphs[l].surface
                        for l in range(1, len(path_ele_of_j)-1):
                            X_list.append(path_ele_of_j[l][0])
                        for l in range(1, len(path_ele_of_k)-1):
                            Y_list.append(path_ele_of_k[l][0])
                        print(" -> ".join(X_list), "|", " -> ".join(Y_list), "|", path_ele_of_j[len(path_ele_of_j)-1][0])

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
