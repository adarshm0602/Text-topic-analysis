import os
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2, mutual_info_classif, SelectKBest
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

# Load labels.csv
df_labels = pd.read_csv("labels.csv")

# Check label counts before processing
label_distribution = df_labels['label'].value_counts()
print("\n📊 Label Distribution before filtering:\n", label_distribution)

texts = []
labels = []

# Read files from 'posts_all/' folder
for _, row in df_labels.iterrows():
    filename = row['filename']
    label = row['label']
    try:
        with open(os.path.join("posts_all", filename), 'r', encoding='utf-8') as file:
            texts.append(file.read())
            labels.append(label)
    except FileNotFoundError:
        print(f"⚠️ File not found: {filename}. Skipping.")

# Exit if no files loaded
if not texts:
    print("❌ No text files loaded. Please check your posts_all folder and labels.csv.")
    exit()

# Remove labels with fewer than 2 samples
label_counts = Counter(labels)
filtered = [(t, l) for t, l in zip(texts, labels) if label_counts[l] >= 2]

# Exit if nothing remains after filtering
if not filtered:
    print("❌ No labels have at least 2 samples. Please check your dataset.")
    exit()

texts, labels = zip(*filtered)

print(f"\n✅ Total usable samples after filtering: {len(texts)}\n")

# Vectorize text data using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(texts)
y = labels

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature selection functions
def select_features_chi2(X_train, y_train, X_test, k=300):
    selector = SelectKBest(chi2, k=min(k, X_train.shape[1]))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    return X_train_sel, X_test_sel

def select_features_mutual_info(X_train, y_train, X_test, k=300):
    selector = SelectKBest(mutual_info_classif, k=min(k, X_train.shape[1]))
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    return X_train_sel, X_test_sel

def select_features_l1_logistic(X_train, y_train, X_test, C=1.0):
    model = LogisticRegression(penalty='l1', solver='liblinear', C=C, max_iter=1000)
    model.fit(X_train, y_train)
    coef = model.coef_
    selected = np.any(coef != 0, axis=0)
    num_selected = np.sum(selected)
    print(f"📌 Selected {num_selected} features via L1 Logistic Regression")

    if num_selected == 0:
        return None, None, 0

    X_train_sel = X_train[:, selected]
    X_test_sel = X_test[:, selected]
    return X_train_sel, X_test_sel, num_selected

# Evaluation function
def evaluate_model(X_train_sel, X_test_sel, y_train, y_test):
    model = MultinomialNB()
    model.fit(X_train_sel, y_train)
    y_pred = model.predict(X_test_sel)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred))

# Chi-Square Feature Selection
print("\n🔍 Feature Selection: Chi-Square")
X_train_sel, X_test_sel = select_features_chi2(X_train, y_train, X_test)
evaluate_model(X_train_sel, X_test_sel, y_train, y_test)

# Mutual Information Feature Selection
print("\n🔍 Feature Selection: Mutual Information")
X_train_sel, X_test_sel = select_features_mutual_info(X_train, y_train, X_test)
evaluate_model(X_train_sel, X_test_sel, y_train, y_test)

# L1-based Logistic Regression Feature Selection
print("\n🔍 Feature Selection: L1-Based Logistic Regression")
X_train_sel, X_test_sel, num_selected = select_features_l1_logistic(X_train, y_train, X_test, C=1.0)

if num_selected == 0:
    print("❌ No features selected via L1 Logistic Regression. Skipping evaluation.")
else:
    evaluate_model(X_train_sel, X_test_sel, y_train, y_test)

print("\n✅ Feature selection experiment completed.\n")
