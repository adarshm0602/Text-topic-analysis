import os
import re
from tabulate import tabulate
from capitalize_method import extract_description
from text_analyzer import extract_topic, clean_text
from stopwords_processor import load_stopwords
from tfidf_manual import compute_tfidf


def read_correct_answer(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        first_line = file.readline().strip()
        match = re.match(r"^[#/]\s*(.*)", first_line)
        return match.group(1) if match else None


def evaluate_methods(folder_path):
    if not os.path.exists(folder_path):
        print(f"Folder '{folder_path}' does not exist.")
        return

    results = []
    stopwords = load_stopwords('./stopwords')

    file_list = [
        f for f in os.listdir(folder_path)
        if f.endswith(".txt") and f[:-4].isdigit() and 1 <= int(f[:-4]) <= 120
    ]  # Limit to 120 files

    if not file_list:
        print("No valid text files found in the folder.")
        return

    for file_name in file_list:
        try:
            file_path = os.path.join(folder_path, file_name)
            correct_answer = read_correct_answer(file_path)

            if not correct_answer:
                continue

            with open(file_path, 'r', encoding='utf-8') as file:
                raw_text = file.read()

            processed_text = clean_text(raw_text)

            # Pass stopwords to extract_description
            capitalize_output = extract_description(raw_text, stopwords)
            tfidf_scores = compute_tfidf(processed_text, stopwords)
            tfidf_output = extract_topic(tfidf_scores)

            capitalize_score = 1 if any(
                word.lower() in correct_answer.lower().split()
                for word in capitalize_output.lower().split()
            ) else 0

            tfidf_score = 1 if any(
                word.lower() in correct_answer.lower().split()
                for word in tfidf_output.lower().split()
            ) else 0

            results.append(["Capitalize Method", file_name, capitalize_score])
            results.append(["TF-IDF Method", file_name, tfidf_score])

        except Exception as e:
            print(f"\nError processing {file_name}: {str(e)}")
            continue

    if not results:
        print("\nNo results to display - no valid labeled files were processed.")
        return

    # Calculate results without pandas
    method_scores = {"Capitalize Method": [], "TF-IDF Method": []}
    for result in results:
        method_scores[result[0]].append(result[2])

    summary_table = []
    for method, scores in method_scores.items():
        total_score = sum(scores)
        accuracy = (total_score / len(scores)) * 100 if scores else 0
        summary_table.append([method, total_score, f"{accuracy:.2f}"])

    print("\n--- Accuracy Evaluation ---")
    print(tabulate(summary_table, headers=["Method Name", "Total Score", "Accuracy (%)"], tablefmt="grid"))


if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing text files: ").strip()
    evaluate_methods(folder_path)
