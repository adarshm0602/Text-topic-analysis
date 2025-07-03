import os
from tabulate import tabulate
from capitalization import extract_description
from text_analyzer import clean_text, extract_topic
from stopwords_processor import load_stopwords
from tfidf_manual import compute_tfidf

def extract_expected_topic(line):
    if line.startswith("//") or line.startswith("#"):
        return line[2:].strip().lower()
    return ""

def predict_topic_capitalization(text):
    return extract_description(text)

def predict_topic_tfidf(text):
    try:
        stopwords = load_stopwords('./stopwords')
    except FileNotFoundError:
        print("Warning: Stopwords folder not found. Continuing without stopwords.")
        stopwords = set()

    words = clean_text(text)
    tfidf_scores = compute_tfidf(words, stopwords)
    return extract_topic(tfidf_scores)

def evaluate_methods(folder_path="posts"):
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist")
        return
        
    files = [f for f in os.listdir(folder_path) if f.endswith(".txt")]
    total_files = len(files)
    correct_cap = 0
    correct_tfidf = 0

    for file in files:
        path = os.path.join(folder_path, file)
        try:
            with open(path, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            if not lines:
                continue  # skip completely empty files

            # Determine expected topic and content based on number of lines
            if lines[0].startswith("//") or lines[0].startswith("#"):
                expected = extract_expected_topic(lines[0])
                content = "".join(lines[1:]) if len(lines) > 1 else ""
            else:
                expected = ""  # no expected topic
                content = "".join(lines)

            # Skip if content is empty
            if not content.strip():
                continue

            predicted_cap = predict_topic_capitalization(content).lower()
            predicted_tfidf = predict_topic_tfidf(content).lower()

            if expected:
                if expected in predicted_cap:
                    correct_cap += 1
                if expected in predicted_tfidf:
                    correct_tfidf += 1

        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue

    if total_files == 0:
        print("No valid text files found in the folder.")
        return

    acc_cap = (correct_cap / total_files) * 100
    acc_tfidf = (correct_tfidf / total_files) * 100

    table = [
        ["Capitalization", f"{acc_cap:.2f}%"],
        ["TF-IDF", f"{acc_tfidf:.2f}%"]
    ]

    print(tabulate(table, headers=["Method", "Accuracy"], tablefmt="grid"))

if __name__ == "__main__":
    evaluate_methods()

