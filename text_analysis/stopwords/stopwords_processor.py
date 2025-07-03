import os

def load_stopwords(stopwords_folder):
    stopwords = set()
    for filename in os.listdir(stopwords_folder):
        filepath = os.path.join(stopwords_folder, filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            for line in file:
                word = line.strip().lower()
                if word:
                    stopwords.add(word)
    return stopwords
