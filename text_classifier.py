import os
import pandas as pd
import collections
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# 📂 Paths
folder_path = 'posts_all/'
labels_file = 'labels.csv'

# 📥 Load labels CSV
df_labels = pd.read_csv(labels_file)

# 📖 Read texts and clean labels
texts, labels = [], []

for _, row in df_labels.iterrows():
    file_path = os.path.join(folder_path, row['filename'])
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        content = ''.join(lines[1:]).strip()  # skip label line
        texts.append(content)

        raw_label = row['label'].strip().lower()

        if raw_label in ['random msg', 'random message']:
            clean_label = 'random message'
        elif raw_label == 'general message.':
            clean_label = 'general message'
        else:
            clean_label = raw_label

        labels.append(clean_label)

print(f"\n✅ Loaded {len(texts)} documents.")

# 📊 Label distribution
label_counts = collections.Counter(labels)
print("\nLabel counts before filtering:", label_counts)

# 🔍 Remove labels with < 2 samples
valid_labels = [label for label, count in label_counts.items() if count >= 2]

texts_filtered = [text for text, label in zip(texts, labels) if label in valid_labels]
labels_filtered = [label for label in labels if label in valid_labels]

filtered_counts = collections.Counter(labels_filtered)
print("\nLabel counts after filtering:", filtered_counts)
print(f"\n✅ Remaining {len(labels_filtered)} documents after filtering.")

# 📝 TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(texts_filtered)

# 📊 Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels_filtered, test_size=0.2, random_state=42, stratify=labels_filtered
)

# 🧠 Train Naive Bayes Classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# 📊 Predict and Evaluate
y_pred = model.predict(X_test)

print("\n✅ Model Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# 📑 Function to predict topic for a new file
def predict_topic(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        content = ''.join(lines[1:]).strip()
    vector = vectorizer.transform([content])
    prediction = model.predict(vector)
    return prediction[0]

# 🔍 Example Usage:
new_file = 'posts_all/some_new_file.txt'
if os.path.exists(new_file):
    result = predict_topic(new_file)
    print(f"\nPredicted Topic for {new_file}: {result}")
else:
    print(f"\nNew file '{new_file}' not found.")
