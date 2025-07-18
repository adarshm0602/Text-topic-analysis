import os
import re
import string
from fpdf import FPDF
from Capitalize_method import extract_description  
from stopwords_processor import load_stopwords  

class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(200, 10, "First Sentence Using Capitalized Method Analysis Report", ln=True, align="C")
        self.ln(10)

    def add_text(self, text):
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 10, text)
        self.ln()

def extract_first_sentence(text):
    text = text.strip()
    lines = text.splitlines()
    first_comment = None

    if lines and (lines[0].startswith("//") or lines[0].startswith("#")):
        first_comment = lines[0].lstrip("//#").strip()
        lines.pop(0)  

    cleaned_text = ' '.join(lines)
    sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)

    for sentence in sentences:
        stripped = sentence.strip(string.whitespace)
        if stripped:
            return first_comment, stripped
    return first_comment,""

def compute_match_score(comment, topic):
    if not comment or not topic:
        return 0  

    comment_words = set(comment.lower().translate(str.maketrans("", "", string.punctuation)).split())
    topic_words = set(topic.lower().translate(str.maketrans("", "", string.punctuation)).split())

    return 1 if comment_words.intersection(topic_words) else 0  

def clean_text(text):
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')  
    text = text.replace("–", "-")  
    return text.encode("latin-1", "ignore").decode("latin-1") 

folder_path = input("Enter the folder path containing text files: ").strip()
if not os.path.isdir(folder_path):
    print("Invalid folder path. Please check and try again.")
    exit()

stopwords = load_stopwords('./stopwords')  
report_data = []
processed_count = 0
match_count = 0

files = sorted([f for f in os.listdir(folder_path) if f.endswith(".txt")])[:120]

for filename in files:
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
        first_comment, first_sentence = extract_first_sentence(content)

        if not first_sentence:
            continue

        topic = extract_description(first_sentence, stopwords)
        match_score = compute_match_score(first_comment, topic) if first_comment else 0
        match_count += match_score
        processed_count += 1
        report_data.append((filename, first_sentence, topic, match_score))

accuracy = (match_count / max(processed_count, 1)) * 100
print(f"\nTotal Files Processed: {processed_count}")
print(f"Accuracy: {accuracy:.2f}%")
pdf = PDFReport()
pdf.add_page()
pdf.add_text(clean_text(f"Capitalized Description Analysis Report\n"))
pdf.add_text(clean_text(f"Total Files Processed: {processed_count}"))
pdf.add_text(clean_text(f"Accuracy: {accuracy:.2f}%\n"))

for filename, sentence, topic, match_score in report_data:
    pdf.add_text(clean_text(f"------------------------------------"))
    pdf.add_text(clean_text(f"File: {filename}"))
    pdf.add_text(clean_text(f"First Sentence: {sentence}"))
    pdf.add_text(clean_text(f"Extracted Topic: {topic}"))
    pdf.add_text(clean_text(f"Match Score: {match_score}"))
    pdf.add_text(clean_text(f"------------------------------------\n"))

pdf.add_text(clean_text(f"Summary:\n- Files Analyzed: {processed_count}"))
pdf.add_text(clean_text(f"- Matched Files: {match_count}"))
pdf.add_text(clean_text(f"- Accuracy: {accuracy:.2f}%\n"))
pdf.add_text(clean_text(f"Analysis completed. Report saved as Capitalized_Description_Report.pdf"))

pdf_filename = "FSC_report.pdf"
pdf.output(pdf_filename, "F")  
print(f"Analysis report saved as {pdf_filename}")
