import os
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 📂 Paths
folder_path = 'posts_all/'
labels_file = 'labels.csv'
report_file = 'classification_report.md'

# 📥 Load labels
df_labels = pd.read_csv(labels_file)

# 📋 Validate all file paths before proceeding
missing_files = []
for _, row in df_labels.iterrows():
    file_path = os.path.join(folder_path, row['filename'])
    if not os.path.exists(file_path):
        missing_files.append(file_path)

if missing_files:
    print("\n❌ Missing files detected:")
    for f in missing_files:
        print(f)
    exit()

print("\n✅ All files found. Proceeding...\n")

# 📖 Read texts and labels
texts, labels = [], []

for _, row in df_labels.iterrows():
    file_path = os.path.join(folder_path, row['filename'])
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        content = ''.join(lines[1:]).strip()  # Skip label line
        texts.append(content)
        labels.append(row['label'])

print(f"\n✅ Loaded {len(texts)} documents.")

# 📊 Check label counts
label_counts = Counter(labels)
print("\nLabel counts before filtering:", label_counts)

# 🔍 Filter out labels with < 2 samples
filtered_texts, filtered_labels = [], []
for text, label in zip(texts, labels):
    if label_counts[label] >= 2:
        filtered_texts.append(text)
        filtered_labels.append(label)

print(f"\n✅ Remaining {len(filtered_texts)} documents after filtering.")

# 📝 Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(filtered_texts)

# 📊 Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, filtered_labels, test_size=0.2, random_state=42, stratify=filtered_labels
)

# 📦 Define models
models = {
    "Naive Bayes": MultinomialNB(),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "K-Nearest Neighbors (k=3)": KNeighborsClassifier(n_neighbors=3)
}

# 📑 Open report file
with open(report_file, 'w', encoding='utf-8') as f:
    f.write("# 📊 Text Classification Model Report\n\n")
    f.write(f"**Total documents used:** {len(filtered_texts)}\n\n")
    f.write("**Train-Test Split:** 80-20 Stratified\n\n")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = round(accuracy_score(y_test, y_pred) * 100, 2)
        clf_report = classification_report(y_test, y_pred)

        print(f"\n✅ {name} Evaluation:")
        print(f"Accuracy: {acc}%")
        print("Classification Report:\n", clf_report)

        # 📄 Write to markdown file
        f.write(f"## ✅ {name}\n")
        f.write(f"**Accuracy:** {acc}%\n\n")
        f.write("**Classification Report:**\n\n")
        f.write("```\n")
        f.write(clf_report)
        f.write("\n```\n\n")

print(f"\n✅ Report generated successfully: {report_file}")
