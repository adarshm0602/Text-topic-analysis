import os
import re
import string
import pandas as pd
from fpdf import FPDF
from Capitalize_method import extract_description
from tfidf import extract_topic, clean_text
from stopwords_processor import load_stopwords
from tfidf_manual import compute_tfidf
from Significant_sentence import *
from tabulate import tabulate


def normalize_text(text):
    """Normalize text for comparison."""
    if not text:
        return ""
    text = text.lower().replace("-", " ").strip()
    text = re.sub(r"[^\w\s]", "", text)  
    text = re.sub(r"\s+", " ", text)  
    return text

def calculate_match_score(reference_text, extracted_text):
    """Calculate similarity score based on matching words."""
    if not reference_text or not extracted_text:
        return 0

    ref_words = set(normalize_text(reference_text).split())
    ext_words = set(normalize_text(extracted_text).split())

    return 1 if ref_words & ext_words else 0 

def read_correct_answer(file_path):
    """Extract commented line from text file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        first_line = file.readline().strip()
        match = re.match(r"^[#/]\s*(.*)", first_line)
        return match.group(1) if match else None

def clean_unicode(text):
    """Remove non-ASCII characters to avoid encoding errors in PDFs."""
    return text.encode("ascii", "ignore").decode()

def extract_first_sentence(text):
    """Extract the first sentence from the text."""
    text = text.strip()
    lines = text.splitlines()
    non_comment_lines = [line for line in lines if not line.strip().startswith(("#", "//"))]
    cleaned_text = ' '.join(non_comment_lines)
    sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)

    for sentence in sentences:
        stripped = sentence.strip(string.whitespace)
        if stripped:
            return stripped
    return ""

def extract_last_sentence(text):
    """Extract the last valid sentence from the text."""
    text = text.strip()
    lines = text.splitlines()
    non_comment_lines = [line for line in lines if not line.strip().startswith(("#", "//"))]
    cleaned_text = ' '.join(non_comment_lines)
    sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)

    for sentence in reversed(sentences):
        stripped = sentence.strip(string.whitespace)
        if stripped:
            return stripped
    return ""
def extract_significant_sentence(text, stopwords):
    """Extract the sentence with the highest TF-IDF score."""
    sentences = split_into_sentences(text)
    if not sentences:
        return ""

    sentence_scores = compute_tfidf_manual(sentences, stopwords)
    significant_sentence = max(sentence_scores, key=lambda x: x[1])[0]
    return significant_sentence


def generate_pdf_report(results, output_file="evaluation_report.pdf"):
    """Generate structured PDF report with extracted descriptions, comments, and scores."""
    if not results:
        print("No results to generate a report.")
        return

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    pdf.cell(0, 10, "TF-IDF & Capitalization Method Evaluation Report", ln=True, align="C")
    pdf.cell(0, 10, "-" * 80, ln=True)

    for result in results:
        pdf.set_font("Arial", style='B', size=12)
        pdf.cell(0, 10, f"File: {result['File Name']}", ln=True)
        pdf.set_font("Arial", size=12)

        pdf.multi_cell(0, 10, f"Commented Line: {clean_unicode(result['Commented Line'])}")
        pdf.multi_cell(0, 10, f"Capitalize Method Description: {clean_unicode(result['Capitalize Method Description'])}")
        pdf.multi_cell(0, 10, f"TF-IDF Method Description: {clean_unicode(result['TF-IDF Method Description'])}")
        pdf.multi_cell(0, 10, f"First Sentence Capitalize Description: {clean_unicode(result['First Sentence Capitalize Description'])}")
        pdf.multi_cell(0, 10, f"First Sentence TF-IDF Description: {clean_unicode(result['First Sentence TF-IDF Description'])}")
        pdf.multi_cell(0, 10, f"Last Sentence Capitalize Description: {clean_unicode(result['Last Sentence Capitalize Description'])}")
        pdf.multi_cell(0, 10, f"Last Sentence TF-IDF Description: {clean_unicode(result['Last Sentence TF-IDF Description'])}")
        pdf.multi_cell(0, 10, f"Significant Sentence Capitalize Description: {clean_unicode(result['Significant Sentence Capitalize Description'])}")
        pdf.multi_cell(0, 10, f"Significant Sentence TF-IDF Description: {clean_unicode(result['Significant Sentence TF-IDF Description'])}")
        

        pdf.cell(0, 10, f"Capitalize Method Score: {result['Capitalize Method Score']}", ln=True)
        pdf.cell(0, 10, f"TF-IDF Method Score: {result['TF-IDF Method Score']}", ln=True)
        pdf.cell(0, 10, f"First Sentence Capitalize Score: {result['First Sentence Capitalize Score']}", ln=True)
        pdf.cell(0, 10, f"First Sentence TF-IDF Score: {result['First Sentence TF-IDF Score']}", ln=True)
        pdf.cell(0, 10, f"Last Sentence Capitalize Score: {result['Last Sentence Capitalize Score']}", ln=True)
        pdf.cell(0, 10, f"Last Sentence TF-IDF Score: {result['Last Sentence TF-IDF Score']}", ln=True)
        pdf.cell(0, 10, f"Significant Sentence Capitalize Score: {result['Significant Sentence Capitalize Score']}", ln=True)
        pdf.cell(0, 10, f"Signifiacnt Sentence TF-IDF Score: {result['Significant Sentence TF-IDF Score']}", ln=True)
        pdf.cell(0, 10, "-" * 80, ln=True)

    pdf.output(output_file)
    print("\nPDF report generated successfully:", output_file)

def evaluate_methods(folder_path):
    """Evaluate multiple extraction methods on text files and display results in the console."""
    stopwords = load_stopwords('./stopwords')
    file_list = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".txt")],
        key=lambda x: x[:-4]
    )

    if not file_list:
        print("No valid text files found in the specified folder.")
        return

    results = []
    total_scores = {
        "Capitalize Method": 0,
        "TF-IDF Method": 0,
        "First Sentence Capitalize": 0,
        "First Sentence TF-IDF": 0,
        "Last Sentence Capitalize": 0,
        "Last Sentence TF-IDF": 0,
        "Significant Sentence Capitalize":0,
        "Significant Sentence TF-IDF":0,
    }
    file_list = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt") and f[:-4].isdigit() and 1 <= int(f[:-4]) <= 120] )


    for file_name in file_list:
        file_path = os.path.join(folder_path, file_name)
        correct_answer = read_correct_answer(file_path) or ""

        with open(file_path, 'r', encoding='utf-8') as file:
            raw_text = file.read()

        processed_text = clean_text(raw_text)
        last_sentence = extract_last_sentence(raw_text)
        first_sentence = extract_first_sentence(raw_text)
        significant_sentence=extract_significant_sentence(raw_text,stopwords)

        extractions = {
            "Capitalize Method": extract_description(raw_text, stopwords),
            "TF-IDF Method": extract_topic(compute_tfidf(processed_text, stopwords)),
            "First Sentence Capitalize": extract_description(first_sentence, stopwords),
            "First Sentence TF-IDF": extract_topic(compute_tfidf(clean_text(first_sentence), stopwords)),
            "Last Sentence Capitalize": extract_description(last_sentence, stopwords),
            "Last Sentence TF-IDF": extract_topic(compute_tfidf(clean_text(last_sentence), stopwords)),
            "Significant Sentence Capitalize":extract_description(significant_sentence, stopwords),
            "Significant Sentence TF-IDF":extract_topic(compute_tfidf(clean_text(significant_sentence),stopwords)),
        }

        results.append({
            "File Name": file_name,
            "Commented Line": correct_answer,
            **{method + " Description": output for method, output in extractions.items()},
            **{method + " Score": calculate_match_score(correct_answer, output) for method, output in extractions.items()},
        })

        for method, output in extractions.items():
            total_scores[method] += calculate_match_score(correct_answer, output)

    generate_pdf_report(results)

    total_files = len(file_list)
    table_data = [
        [method, score, f"{(score/total_files)*100:.2f}%"] for method, score in total_scores.items()
    ]
    print("\nEvaluation Summary:")
    print(tabulate(table_data, headers=["Method", "Total Score", "Accuracy"], tablefmt="grid"))

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing text files: ").strip()
    evaluate_methods(folder_path)