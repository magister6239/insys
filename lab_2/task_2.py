from math import log
import numpy as np


class NaiveBayesClassifier:
    def __init__(self):
        self.texts = dict()
        self.dictionary = set()
        self.class_probabilities = dict()

    def add_text(self, text: list, k):
        if k not in self.texts:
            self.texts[k] = list()

        self.texts[k].append(text)

        self.dictionary.update(text)

    def get_bow_vectors(self):
        vectors = []

        for text in self.texts.values():

            vector = []

            for word in self.dictionary:
                if word in text:
                    vector.append(1)
                else:
                    vector.append(0)

            vectors.append(vector)

        return np.array(vectors)

    def get_tfidf_vectors(self):
        vectors = []

        for text in self.texts.values():

            vector = []

            for word in self.dictionary:
                tfidf = self.__calculate_tf(word, text) * self.__calculate_idf(word)
                vector.append(tfidf)

            vectors.append(vector)

        return np.array(vectors)

    @staticmethod
    def __calculate_tf(word, text):
        return text.count(word) / len(text)

    def __calculate_idf(self, word):
        t_container_count = 0
        for text in self.texts.values():
            if word in text:
                t_container_count += 1

        return log(len(self.texts.values()) / (t_container_count + 1))

    def train(self):
        total_texts = sum(len(texts) for texts in self.texts.values())
        for class_name, texts in self.texts.items():
            self.class_probabilities[class_name] = len(texts) / total_texts

    def predict(self, text):
        clean_text = str().join(char for char in text.lower() if char.isalpha() or char.isspace())
        words = clean_text.split()

        tfidf_vector = [self.__calculate_tf(word, words) * self.__calculate_idf(word) for word in self.dictionary]

        class_scores = {}
        for class_name in self.class_probabilities:
            class_score = log(self.class_probabilities[class_name])
            for word, tfidf in zip(self.dictionary, tfidf_vector):
                if tfidf > 0:
                    class_score += log(self.__calculate_likelihood(word, class_name)) * tfidf

            class_scores[class_name] = class_score

        return max(class_scores, key=class_scores.get)

    def __calculate_likelihood(self, word, class_name):
        class_texts = self.texts[class_name]
        word_count_in_class = sum(text.count(word) for text in class_texts)
        total_words_in_class = sum(len(text) for text in class_texts)

        return (word_count_in_class + 1) / (total_words_in_class + len(self.dictionary))


n = NaiveBayesClassifier()

with open("SMSSpamCollection", "r", encoding="utf-8") as f:
    for line in f.readlines():
        text = line.lower()
        clean_text = str().join(char for char in text if char.isalpha() or char.isspace())
        words = clean_text.split()
        cl = words.pop(0)
        n.add_text(words, cl)

n.train()

with open("testtt.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        prediction = n.predict(line)
        print(prediction)

