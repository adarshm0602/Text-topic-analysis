import os
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# 📂 Paths
folder_path = 'posts_all/'
labels_file = 'labels.csv'

# 📥 Load labels
df_labels = pd.read_csv(labels_file)

# 📖 Read texts and labels
texts, labels = [], []

for _, row in df_labels.iterrows():
    file_path = os.path.join(folder_path, row['filename'])
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        content = ''.join(lines[1:]).strip()  # Skip label line
        texts.append(content)

        # 🔍 Clean and merge labels
        raw_label = row['label'].strip().lower()

        if raw_label in ['random msg', 'random message']:
            clean_label = 'random message'
        elif raw_label in ['general message.']:
            clean_label = 'general message'
        else:
            clean_label = raw_label

        labels.append(clean_label)

print(f"\n✅ Loaded {len(texts)} documents.")

# 📊 Check and display label counts
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

# 🧠 Train Linear SVM classifier
model = LinearSVC()
model.fit(X_train, y_train)

# 📊 Predict and evaluate
y_pred = model.predict(X_test)

print("\n✅ Linear SVM Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
