import os
import string
from collections import Counter
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize 
from stopwords_processor import load_stopwords

def get_significant_sentences_frequency(text, top_n=1):
    stopwords = load_stopwords('./stopwords')
    
    sentences = sent_tokenize(text)
    words = [word.lower() for word in word_tokenize(text) if word.lower() not in stopwords]
    word_freq = Counter(words)

    sentence_scores = {}
    for sentence in sentences:
        sentence_words = [word.lower() for word in word_tokenize(sentence) if word.lower() not in stopwords]
        score = sum(word_freq.get(word, 0) for word in sentence_words)
        sentence_scores[sentence] = score

    significant_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:top_n]
    return significant_sentences

folder_name = input("Enter the folder name (e.g. textfiles): ").strip()
filename = input("Enter the filename with extension (e.g. file1.txt): ").strip()

filepath = os.path.join('.', folder_name, filename)

if os.path.exists(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    # Filter out lines starting with // or #
    valid_lines = [line for line in lines if not line.strip().startswith(("//", "#"))]

    # Combine remaining lines into a single text
    text = ' '.join(valid_lines)

    top_sentences = get_significant_sentences_frequency(text)

    print("\nTop Significant Sentence (Frequency-based):")
    for sentence in top_sentences:
        print(f"- {sentence}")

else:
    print("File not found. Please check the folder and file name and try again.")
