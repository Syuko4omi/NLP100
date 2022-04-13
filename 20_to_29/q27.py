import json
import re

decoder = json.JSONDecoder()

Ingland_text = None
pat_1 = re.compile(r"^\{\{基礎情報.*?$(.*)^\}\}$", re.MULTILINE + re.DOTALL)
pat_2 = re.compile(r'''^\|(.*?)\s*=\s*(.*?)
                    (?:
                        (?=\n\|)|(?=\n$) #改行直後にまた続くか、一番最後のフィールド
                    )
                        ''', re.MULTILINE + re.DOTALL + re.VERBOSE)

def pickup_Ingland():
    with open("jawiki-country.json") as file_1:
        Lines_jsonStyle = file_1.readlines()

        for i in range(len(Lines_jsonStyle)):
            Line_dictStyle = decoder.decode(Lines_jsonStyle[i])
            if Line_dictStyle["title"] == "イギリス":
                return Line_dictStyle["text"]

Ingland_text = pickup_Ingland()

basic_info_list = pat_1.findall(Ingland_text)
field_and_val_list = pat_2.findall(basic_info_list[0])

ans_dict = {}
for i in range(len(field_and_val_list)):
    field = field_and_val_list[i][0]
    val = re.sub(r"'{2,5}", r"", field_and_val_list[i][1])
    remove_internal_link = val
    if not "ファイル" in val:
        #[[hoge]]もしくは[[fuga|hoge]]をhogeにする。
        #[[A]]hogehoge[[B|C]]みたいなのが[[A|C]]になるのを防ぐために、|の前は[]を含まないものとして定義
        remove_internal_link = re.sub(r"\[\[([^\[|^\]]*?\|)?(.*?)\]\]", r"\2", val)
    ans_dict[field] = remove_internal_link
    print(field, remove_internal_link)
