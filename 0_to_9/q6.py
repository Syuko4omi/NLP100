def make_bigram_list(word):
    bi_gram_list = []
    for i in range(len(word)-1):
        bi_gram_list.append(word[i:i+2])
    return bi_gram_list

X = list(set(make_bigram_list("paraparaparadise")))
Y = list(set(make_bigram_list("paragraph")))
print(X)
print(Y)

Union = list(set(X+Y))
Intersection = [X[i] for i in range(len(X)) if X[i] in Y]
Difference = [Union[i] for i in range(len(Union)) if Union[i] not in Intersection]
print(Union)
print(Intersection)
print(Difference)

print("se" in Union)
