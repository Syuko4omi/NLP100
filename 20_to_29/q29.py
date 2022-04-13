import requests
import json
import re
#import webbrowser

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
    remove_internal_link = re.sub(r"\[\[([^\[|^\]]*?\|)*(.*?)\]\]", r"\2", val)
    remove_template_lang = re.sub(r"\**\{\{lang\|.*?\|(.*?)\}\}", r"\1", remove_internal_link)
    ans_dict[field] = remove_template_lang


flag_photo_filename = ans_dict["国旗画像"]
S = requests.Session()
URL = "https://en.wikipedia.org/w/api.php"
PARAMS = {
    "action": "query",
    "format": "json",
    "prop": "imageinfo",
    "titles": "File:"+flag_photo_filename,
    "iiprop": "url"
}

R = S.get(url=URL, params=PARAMS)
DATA = R.json()

PAGES = DATA["query"]["pages"]
for k, v in PAGES.items():
    print("url:", v["imageinfo"][0]["url"])
    #webbrowser.open(v["imageinfo"][0]["url"])
