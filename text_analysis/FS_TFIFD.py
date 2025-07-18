import os
import re
import string
from fpdf import FPDF
from tfidf_manual import compute_tfidf
from stopwords_processor import load_stopwords

class PDFReport(FPDF):
    def header(self):
        self.set_font("Arial", "B", 14)
        self.cell(200, 10, "First Sentence Using TF-IDF Report", ln=True, align="C")
        self.ln(10)

    def add_text(self, text):
        self.set_font("Arial", "", 12)
        self.multi_cell(0, 10, text)
        self.ln()

def extract_first_sentence(text):
    """Extract the first valid sentence and keep the commented line for comparison."""
    text = text.strip()
    lines = text.splitlines()
    first_comment = None


    for line in lines:
        if line.strip().startswith("//") or line.strip().startswith("#"):
            first_comment = line.lstrip("//#").strip()
            break  


    non_comment_lines = [line for line in lines if not line.strip().startswith("//") and not line.strip().startswith("#")]
    cleaned_text = ' '.join(non_comment_lines)


    sentences = re.split(r'(?<=[.!?])\s+', cleaned_text)
    for sentence in sentences:
        stripped = sentence.strip(string.whitespace)
        if stripped:
            return first_comment, stripped

    return first_comment, ""  

def compute_match_score(comment, topic):
    """Assigns a score of `1` if at least one word matches between the commented line and detected topic."""
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

file_list = sorted(
        [f for f in os.listdir(folder_path) if f.endswith(".txt") and f[:-4].isdigit()],
        key=lambda x: int(x[:-4])
    )[:120]  


for index, filename in enumerate(file_list, 1):  
    file_path = os.path.join(folder_path, filename)
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
        content = file.read()
        first_comment, first_sentence = extract_first_sentence(content)

        if not first_sentence:
            continue


        words = first_sentence.lower().translate(str.maketrans("", "", string.punctuation)).split()
        tfidf_scores = compute_tfidf(words, stopwords)

        sorted_words = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)  
        top_words = [word for word, score in sorted_words[:3] if not word.isnumeric()]  
        topic = " ".join(top_words) if top_words else "Could not determine a clear topic."

        match_score = compute_match_score(first_comment, topic) if first_comment else 0
        match_count += match_score
        processed_count += 1

        report_data.append((index, filename, first_sentence, topic, match_score))

accuracy = (match_count / max(processed_count, 1)) * 100
print(f"\nTotal Files Processed: {processed_count}")
print(f"Accuracy: {accuracy:.2f}%")

pdf = PDFReport()
pdf.add_page()
pdf.add_text("First Sentence Using TF-IDF Report\n")
pdf.add_text(f"Total Files Processed: {processed_count}")
pdf.add_text(f"Accuracy: {accuracy:.2f}%\n")

for index, filename, sentence, topic, match_score in report_data:
    pdf.add_text(f"{index}. File: {filename}")
    pdf.add_text(f"   First Sentence: {clean_text(sentence)}")
    pdf.add_text(f"   Extracted Topic: {clean_text(topic)}")
    pdf.add_text(f"   Match Score: {match_score}\n")

pdf.add_text("Analysis completed. Report saved as First_Sentence_TFIDF_Report.pdf")

pdf_filename = "FS_TFIDF_Report.pdf"
pdf.output(pdf_filename, "F")  
print(f" analysis report saved as {pdf_filename}")