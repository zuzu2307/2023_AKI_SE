def search(text, patt):
    patt_lenght = len(patt)
    text_lenght = len(text)
    check = 0
    hash_patt = 0
    hash_text_window = 0
    for i in patt:
        check += 1
        hash_patt += (ord(i)-96)*(pow(10, patt_lenght-check))

    for i in range(text_lenght - patt_lenght+1):
        if i < text_lenght - patt_lenght:
            check = 0
            hash_text_window = 0

            for j in range(patt_lenght):
                check += 1
                hash_text_window += (ord(text[j+i])-96) * \
                    (pow(10, patt_lenght-check))

        if hash_text_window == hash_patt:
            for j in range(patt_lenght):
                if text[i+j] != patt[j]:
                    break
                else:
                    j += 1

            if j == patt_lenght:
                print("Pattern found at index " + str(i))


text = 'abbacdadbhabbagpfoe'
patt = 'abba'
search(text, patt)
