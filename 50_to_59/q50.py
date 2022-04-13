import random

valid_publisher_name = ["Reuters", "Huffington Post", "Businessweek", "Contactmusic.com", "Daily Mail"]
with open("NewsAggregatorDataset/newsCorpora.csv", "r") as f:
    valid_data_source = []
    L = f.readlines()
    for i in range(len(L)):
        tab_counter = 0
        publisher = ""
        title = ""
        category = ""
        for j in range(len(L[i])):
            if L[i][j] == "\t":
                tab_counter += 1
            elif tab_counter == 1:
                title += L[i][j]
            elif tab_counter == 3:
                publisher += L[i][j]
            elif tab_counter == 4:
                category += L[i][j]

            if tab_counter > 4:
                break
        if publisher in valid_publisher_name:
            valid_data_source.append(category + "\t" + title + "\n")

    random.shuffle(valid_data_source)
    train_len = int(len(valid_data_source)*0.8)
    valid_len = int(len(valid_data_source)*0.9)

    with open("train.txt", "w") as wf_1:
        wf_1.writelines(valid_data_source[:train_len])
    with open("valid.txt", "w") as wf_2:
        wf_2.writelines(valid_data_source[train_len:valid_len])
    with open("test.txt", "w") as wf_3:
        wf_3.writelines(valid_data_source[valid_len:])
