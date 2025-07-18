import string
import re
from stopwords_processor import load_stopwords
from tfidf_manual import compute_tfidf

def clean_text(text):
    lines = text.split("\n")
    if lines and (lines[0].startswith("//") or lines[0].startswith("#")):
        lines.pop(0)

    text = "\n".join(lines).lower()
    
    text = re.sub(r"[’']", "", text) 
    
    text = text.translate(str.maketrans("", "", string.punctuation + string.digits))
    
    return text.split()

def extract_topic(tfidf_scores):
    sorted_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)  
    top_words = [word for word, score in sorted_words[:3] if not word.isnumeric()] 
    
    if not top_words:
        return "Could not determine a clear topic."
    
    topic = " ".join(top_words)
    
    return topic

def main():
    path = input("Enter the full path to the .txt file: ").strip()
    with open(path, 'r', encoding='utf-8') as file:
        raw_text = file.read()

    stopwords = load_stopwords('./stopwords') 
    words = clean_text(raw_text)  
    
    tfidf_scores = compute_tfidf(words, stopwords)  
    
    topic = extract_topic(tfidf_scores)  


    print("\n--- Analysis Result ---")
    print(topic)
    print("\nTop TF-IDF values:")
    for word, score in sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{word}: {score:.4f}")

if __name__ == "__main__":
    main()