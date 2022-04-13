import json

decoder = json.JSONDecoder()
#JSONのobject -> PythonのDict, JSONのstring -> Pythonのstr, ...etc

with open("jawiki-country.json") as file_1:
    Lines_jsonStyle = file_1.readlines()

    for i in range(len(Lines_jsonStyle)):
        Line_dictStyle = decoder.decode(Lines_jsonStyle[i]) #各行のobjectをDictに変換している
        if Line_dictStyle['title'] == 'イギリス':
            print(Line_dictStyle['text'])
