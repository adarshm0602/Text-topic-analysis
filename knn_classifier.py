import os
import pandas as pd
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

# 📂 Paths
folder_path = 'posts_all/'
labels_file = 'labels.csv'

# 📥 Load labels
df_labels = pd.read_csv(labels_file)

# 📖 Read texts and labels
texts, labels = [], []
for _, row in df_labels.iterrows():
    file_path = os.path.join(folder_path, row['filename'])
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            content = ''.join(lines[1:]).strip()  # Skip label line
            texts.append(content)
            labels.append(row['label'])

# 📊 Filter out labels with < 2 samples
label_counts = Counter(labels)
filtered_texts, filtered_labels = [], []
for text, label in zip(texts, labels):
    if label_counts[label] >= 2:
        filtered_texts.append(text)
        filtered_labels.append(label)

# 📝 Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(filtered_texts)

# 📊 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, filtered_labels, test_size=0.2, random_state=42, stratify=filtered_labels
)

# ✅ Train K-Nearest Neighbors Classifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)

# 📊 Predict & Evaluate
y_pred = model.predict(X_test)
accuracy = round(accuracy_score(y_test, y_pred) * 100, 2)
print(f"\n✅ K-Nearest Neighbors (k=3) Accuracy: {accuracy}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred))
