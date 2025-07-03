import string
import re
from stopwords_processor import load_stopwords
from tfidf_manual import compute_tfidf

def clean_text(text):
    lines = text.split("\n")
    if lines and (lines[0].startswith("//") or lines[0].startswith("#")):
        lines = lines[1:]  # Remove comment line

    text = "\n".join(lines).lower()
    text = re.sub(r'\d+', '', text)  # Remove digits
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text.split()

def extract_topics(tfidf_scores):
    sorted_items = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
    top_words = [word for word, score in sorted_items[:6]]  # Get top 6 words instead of 5

    # Check if there are combinations of words that make sense
    topics = []
    i = 0
    while i < len(top_words) - 1:
        topics.append(f"{top_words[i]} {top_words[i+1]}")  # Combine adjacent words
        i += 2

    # Deduplicate topics (in case the same combination appears more than once)
    topics = list(dict.fromkeys(topics))

    # Heuristic: check if top words are nearly equal in TF-IDF (e.g., this would indicate a general message)
    scores = [score for _, score in sorted_items[:6]]
    if max(scores) - min(scores) < 0.01:
        return "a general message"

    if len(topics) == 1:
        return topics[0]
    elif len(topics) == 2:
        return f"{topics[0]} and {topics[1]}"
    else:
        return ", ".join(topics[:2]) + " and others"



def main():
    path = input("Enter the full path to the .txt file: ").strip()
    with open(path, 'r', encoding='utf-8') as file:
        raw_text = file.read()

    stopwords = load_stopwords('stopwords')
    words = clean_text(raw_text)
    tfidf_scores = compute_tfidf(words, stopwords)

    topic = extract_topics(tfidf_scores)

    print("\n--- Analysis Result ---")
    if topic == "a general message":
        print("This text is a general message.\n")
    else:
        print(f"This text is about {topic}.\n")

    print("Top TF-IDF values:")
    for word, score in sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"{word}: {score:.4f}")

if __name__ == "__main__":
    main()
