def cipher(str):
    encryp = ""
    for i in range(len(str)):
        if ord('a') <= ord(str[i]) <= ord('z'):
            temp = 219-ord(str[i])
            encryp += chr(temp)
        else:
            encryp += str[i]
    return encryp

def anti_cipher(str):
    decryp = ""
    for i in range(len(str)):
        if ord('a') <= -ord(str[i])+219 <= ord('z'):
            decryp += chr(-ord(str[i])+219)
        else:
            decryp += str[i]
    return decryp

print(cipher("Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."))
print(anti_cipher(cipher("Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics.")))
