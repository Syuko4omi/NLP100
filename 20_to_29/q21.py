import json
import re

decoder = json.JSONDecoder()

Ingland_text = None
pat_1 = re.compile(r"^\[\[Category:.*\]\]$", re.MULTILINE)

def pickup_Ingland():
    with open("jawiki-country.json") as file_1:
        Lines_jsonStyle = file_1.readlines()

        for i in range(len(Lines_jsonStyle)):
            Line_dictStyle = decoder.decode(Lines_jsonStyle[i])
            if Line_dictStyle["title"] == "イギリス":
                return Line_dictStyle["text"]

Ingland_text = pickup_Ingland()

ans_list = pat_1.findall(Ingland_text)
for i in range(len(ans_list)):
    print(ans_list[i])
