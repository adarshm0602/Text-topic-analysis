import math
from collections import Counter

def compute_tf(words):
    word_count = Counter(words)
    total_words = len(words)
    return {word: count / total_words for word, count in word_count.items()}

def compute_idf(documents):
    N = len(documents)
    idf = {}
    all_words = set(word for doc in documents for word in doc)
    for word in all_words:
        containing = sum(1 for doc in documents if word in doc)
        idf[word] = math.log((N + 1) / (containing + 1)) + 1 
    return idf

def compute_tfidf(words, stopwords):
    filtered = [w for w in words if w not in stopwords]
    tf = compute_tf(filtered)
    idf = compute_idf([filtered])
    return {word: tf[word] * idf[word] for word in tf}
