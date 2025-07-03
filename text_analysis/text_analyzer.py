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
    top_words = [word for word, score in sorted_words[:6] if not word.isnumeric()]  

    potential_names = []
    multiple_topics_detected = False
    
    for i in range(len(top_words) - 1):
        word1 = top_words[i]
        word2 = top_words[i + 1]
        potential_names.append(f"{word1} {word2}")  


    scores = [score for _, score in sorted_words[:6]]

    
    if max(scores) - min(scores) < 0.0001:  
        return "This text is a general message."

    if len(sorted_words) >= 2 and abs(sorted_words[0][1] - sorted_words[1][1]) > 0.02:
        multiple_topics_detected = True

    if len(top_words) == 1:
        topic = f"This text is about {top_words[0]} (single topic)."
    elif multiple_topics_detected:
        topic = f"This text discusses multiple topics: {', '.join(top_words[:3])}."
    else:
        topic = f"This text is about {potential_names[0]}."

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