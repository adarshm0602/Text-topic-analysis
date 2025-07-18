import re
from collections import Counter
from stopwords_processor import load_stopwords


def extract_description(text, stopwords):
    # Remove comment lines (starting with //)
    lines = [line.strip() for line in text.strip().split('\n') if line.strip() and not line.startswith("//")]
    content = " ".join(lines)

    phrases = re.findall(r'\b(?:[A-Z][a-z]*\s?)*[A-Z][a-z]*\b|\b[A-Z]+\d*[A-Z]*\b', content)

    valid_phrases = []
    for phrase in phrases:
        words = phrase.strip().split()
        # Filter out phrases where all words are stopwords++++++++
        cleaned_words = [w for w in words if w.lower() not in stopwords]
        if cleaned_words and len(" ".join(cleaned_words)) > 3:
            valid_phrases.append(" ".join(cleaned_words))

    # Count frequency of each phrase
    phrase_counts = Counter(valid_phrases)

    if not phrase_counts:
        return "Could not determine a clear topic."

    # Top 3 most frequent phrases
    top_topics = [phrase for phrase, _ in phrase_counts.most_common(3)]
    description = " ".join(top_topics)
    return description


def main():
    while True:
        file_name = input("Enter the file name or type 'exit' to quit: ").strip()
        if file_name.lower() == 'exit':
            break

        try:
            with open(file_name, "r", encoding="utf-8") as file:
                text = file.read()
                stopwords = load_stopwords('./stopwords')  # Load stopwords from folder
                description = extract_description(text, stopwords)
                print(f"\nExtracted Description:\n{description}\n")
        except FileNotFoundError:
            print("File not found. Please check the name and try again.\n")


if __name__ == "__main__":
    main()