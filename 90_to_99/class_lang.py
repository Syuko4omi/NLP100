class Lang:
    def __init__(self, language):
        self.language = language
        self.word2id = {}
        self.id2word = {0: "SOS", 1: "EOS"}
        self.word2count = {}
        self.num_of_words = 2

    def add_word_to_dict(self, word):
        if self.language == "ja":
            if word not in self.word2id:
                if word in ja_words_list:
                    self.word2id[word] = self.num_of_words
                    self.id2word[self.num_of_words] = word
                    self.word2count[word] = 1
                    self.num_of_words += 1
            else:
                self.word2count[word] += 1
        else:
            if word not in self.word2id:
                if word in en_words_list:
                    self.word2id[word] = self.num_of_words
                    self.id2word[self.num_of_words] = word
                    self.word2count[word] = 1
                    self.num_of_words += 1
            else:
                self.word2count[word] += 1

    def add_sentence_to_dict(self, sentence):
        for word in sentence.split():
            self.add_word_to_dict(word)
