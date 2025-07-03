import re
from collections import Counter

IGNORE_WORDS = set([
    "the", "is", "in", "to", "of", "and", "a", "an", "it", "this", "for", "on", "at", "by", "with", 
    "as", "was", "be", "has", "have", "are", "or", "but", "which", "who", "that", "from", "so", "how",
    "their", "its", "he", "she", "his", "her", "we", "you", "i", "they", "them", "not", "just", "now", 
    "then", "could", "should", "would", "may", "might", "also", "into", "about", "over", "under", "all",
    "any", "can", "will", "our", "your", "more", "if", "get", "whether", "while", "new", "launch", "launching",
    "inviting", "invited", "introduced", "introducing", "presenting", "presented", "announcing", "announced","crore"
])

def extract_description(text):
    lines = [line.strip() for line in text.strip().split('\n') if line.strip() and not line.startswith("//") and not line.startswith("#")]
    content = " ".join(lines)

    phrases = re.findall(r'\b(?:[A-Z][a-z]+(?:\s+|$)){1,3}', content)

    valid_phrases = []
    for phrase in phrases:
        words = phrase.strip().split()
        cleaned_words = [w for w in words if w.lower() not in IGNORE_WORDS]
        if cleaned_words and len(" ".join(cleaned_words)) > 3:
            valid_phrases.append(" ".join(cleaned_words))

    phrase_counts = Counter(valid_phrases)

    if not phrase_counts:
        return "Could not determine a clear topic."

    top_topics = [phrase for phrase, _ in phrase_counts.most_common(2)]
    description = "This post describes " + " and ".join(top_topics)
    return description

def main():
    while True:
        file_name = input("Enter the file name  or type 'exit' to quit: ").strip()
        if file_name.lower() == 'exit':
            print("Exiting the program.")
            break
        try:
            with open(file_name, "r", encoding="utf-8") as file:
                text = file.read()
                description = extract_description(text)
                print(f"\nFile: {file_name}")
                print(f"Generated description: {description}")
        except FileNotFoundError:
            print("File not found. Please check the name and try again.")

if __name__ == "__main__":
    main()
