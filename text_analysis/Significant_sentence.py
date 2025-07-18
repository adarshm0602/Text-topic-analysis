import os
import re
import string
import math
from collections import Counter
from fpdf import FPDF
from Capitalize_method import extract_description
from tfidf import extract_topic, clean_text
from stopwords_processor import load_stopwords
from tfidf_manual import compute_tfidf
from tabulate import tabulate

def preprocess_text(text):
    """Clean text by removing special characters and extra spaces."""
    text = text.lower().replace("-", " ").strip()
    text = re.sub(r"[^\w\s]", "", text)  
    text = re.sub(r"\s+", " ", text)  
    return text

def split_into_sentences(text):
    """Split text into individual sentences."""
    return re.split(r'(?<=[.!?])\s+', text.strip())

def compute_tf(sentence):
    """Compute term frequency for a given sentence."""
    words = preprocess_text(sentence).split()
    word_counts = Counter(words)
    total_words = len(words)
    return {word: count / total_words for word, count in word_counts.items()}

def compute_idf(sentences):
    """Compute inverse document frequency for words across all sentences."""
    total_sentences = len(sentences)
    word_document_counts = Counter()

    for sentence in sentences:
        words = set(preprocess_text(sentence).split())
        for word in words:
            word_document_counts[word] += 1

    return {word: math.log(total_sentences / (word_document_counts[word] + 1)) for word in word_document_counts}

def compute_tfidf_manual(sentences, stopwords):
    """Compute TF-IDF scores for each sentence manually."""
    idf_scores = compute_idf(sentences)
    sentence_scores = []

    for sentence in sentences:
        tf_scores = compute_tf(sentence)
        tfidf_score = sum(tf_scores[word] * idf_scores.get(word, 0) for word in tf_scores if word not in stopwords)
        sentence_scores.append((sentence, tfidf_score))

    return sentence_scores

def extract_significant_sentence(text, stopwords):
    """Extract the sentence with the highest TF-IDF score."""
    sentences = split_into_sentences(text)
    if not sentences:
        return ""

    sentence_scores = compute_tfidf_manual(sentences, stopwords)
    significant_sentence = max(sentence_scores, key=lambda x: x[1])[0]
    return significant_sentence

def clean_unicode(text):
    """Convert text to ASCII-friendly format to avoid encoding issues."""
    return text.encode("latin-1", "ignore").decode("latin-1")

def read_correct_answer(file_path):
    """Extract commented line from text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        first_line = file.readline().strip()
        match = re.match(r"^[#/]\s*(.*)", first_line)
        return match.group(1) if match else None

def calculate_match_score(reference_text, extracted_text):
    """Compare reference text with extracted text; return 1 if at least one word matches, otherwise 0."""
    if not reference_text or not extracted_text:
        return 0  # No match possible

    ref_words = set(preprocess_text(reference_text).split())
    ext_words = set(preprocess_text(extracted_text).split())

    return 1 if ref_words & ext_words else 0  

def generate_pdf_report(results, output_file="Signifiacant_report.pdf"):
    """Generate structured PDF report."""
    if not results:
        print("No results to generate a report.")
        return

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", '', 12)

    pdf.cell(0, 10, "Significant Sentence Analysis Report", ln=True, align="C")
    pdf.cell(0, 10, "-" * 80, ln=True)

    for result in results:
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, f"File: {result['File Name']}", ln=True)
        pdf.set_font("Arial", size=12)

        pdf.multi_cell(0, 10, f"Commented Line: {clean_unicode(result['Commented Line'])}")
        pdf.multi_cell(0, 10, f"Significant Sentence: {clean_unicode(result['Significant Sentence'])}")
        pdf.multi_cell(0, 10, f"Capitalization Method Description: {clean_unicode(result['Capitalize Method Description'])}")
        pdf.multi_cell(0, 10, f"TF-IDF Topic Detection: {clean_unicode(result['TF-IDF Topic'])}")
        pdf.multi_cell(0, 10, f"Capitalize Method Score: {result['Capitalize Method Score']}")
        pdf.multi_cell(0, 10, f"TF-IDF Method Score: {result['TF-IDF Method Score']}")

        pdf.cell(0, 10, "-" * 80, ln=True)

    pdf.output(output_file)
    print("\nPDF report generated successfully:", output_file)

def evaluate_methods(folder_path):
    """Evaluate significant sentence extraction, capitalization, and TF-IDF topic detection."""
    stopwords = load_stopwords('./stopwords')
    file_list = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".txt")],
        key=lambda x: x[:-4]
    )[:120]  

    if not file_list:
        print("No valid text files found in the specified folder.")
        return

    results = []
    total_scores = {
        "Capitalize Method": 0,
        "TF-IDF Method": 0
    }
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt") and f[:-4].isdigit()],key=lambda x: int(x[:-4]))[:120]  
    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        correct_answer = read_correct_answer(file_path) or ""

        with open(file_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()

        significant_sentence = extract_significant_sentence(raw_text, stopwords)

        if not significant_sentence:
            continue  
        capitalize_output = extract_description(significant_sentence, stopwords)
        tfidf_output = extract_topic(compute_tfidf(clean_text(significant_sentence), stopwords))

        
        cap_score = calculate_match_score(correct_answer, capitalize_output)
        tfidf_score = calculate_match_score(correct_answer, tfidf_output)

        total_scores["Capitalize Method"] += cap_score
        total_scores["TF-IDF Method"] += tfidf_score

        
        results.append({
            "File Name": file_name,
            "Commented Line": correct_answer,
            "Significant Sentence": significant_sentence,
            "Capitalize Method Description": capitalize_output,
            "TF-IDF Topic": tfidf_output,
            "Capitalize Method Score": cap_score,
            "TF-IDF Method Score": tfidf_score
        })

    generate_pdf_report(results)

    total_files = len(results)
    if total_files > 0:
        print(f"\nProcessed {total_files} files successfully!")
        print("\nEvaluation Summary:")
        table_data = [
            [method, score, f"{(score/total_files)*100:.2f}%"] for method, score in total_scores.items()
        ]
        print(tabulate(table_data, headers=["Method Name", "Score", "Accuracy"], tablefmt="grid"))

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing text files: ").strip()
    evaluate_methods(folder_path)