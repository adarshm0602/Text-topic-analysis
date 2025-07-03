import os
import string
import nltk
from nltk.tokenize import sent_tokenize
from text_analyzer import extract_topic, clean_text
from stopwords_processor import load_stopwords  
from tfidf_manual import compute_tfidf


def get_significant_sentences_tfidf(text, top_n=1):
    sentences = sent_tokenize(text)
    stopwords = load_stopwords('./stopwords')
    
    # Calculate TF-IDF scores for each sentence
    sentence_scores = {}
    for sentence in sentences:
        words = clean_text(sentence)
        tfidf_scores = compute_tfidf(words, stopwords)
        
        # Use average TF-IDF score for the sentence
        if words:
            sentence_scores[sentence] = sum(tfidf_scores.values()) / len(words)
        else:
            sentence_scores[sentence] = 0

    # Sort sentences by score and select top N
    significant_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:top_n]
    return significant_sentences


# Get file path input (folder + filename)
filepath = input("Enter the relative file path (e.g. textfiles/file1.txt): ")

# Check if file exists at the given path
if os.path.exists(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()

    top_sentences = get_significant_sentences_tfidf(text, top_n=2)

    print("\nTop Significant Sentences (TF-IDF-based):")
    for sentence in top_sentences:
        print(f"- {sentence}")

else:
    print("File not found. Please check the file path and try again.")
