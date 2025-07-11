import os
import pandas as pd
import numpy as np
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

import matplotlib.pyplot as plt
import seaborn as sns
import copy

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

print(f"📊 Loaded {len(texts)} documents")
print(f"📊 Label distribution: {Counter(labels)}")

# 📊 Filter labels with at least 2 samples
label_counts = Counter(labels)
filtered_texts, filtered_labels = [], []
for text, label in zip(texts, labels):
    if label_counts[label] >= 2:
        filtered_texts.append(text)
        filtered_labels.append(label)

print(f"📊 After filtering: {len(filtered_texts)} documents")
print(f"📊 Filtered label distribution: {Counter(filtered_labels)}")

y = np.array(filtered_labels)


# 🔁 Universal experiment function
def run_experiment(X_features, y, method_name):
    print(f"\n\n📌 Running experiment with {method_name}")
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5, weights='distance'),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10),
        'Support Vector Machine': SVC(kernel='rbf', random_state=42, probability=True),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Naive Bayes': MultinomialNB(alpha=1.0)
    }

    predictions, scores, cv_means, cv_stds = {}, {}, {}, {}

    for name, model in models.items():
        print(f"🚀 Training {name}")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        scores[name] = accuracy_score(y_test, pred)
        cv = cross_val_score(model, X_train, y_train, cv=5)
        cv_means[name] = cv.mean()
        cv_stds[name] = cv.std()

    # Ensemble
    ensemble = VotingClassifier(
        estimators=[(k.lower().split()[0], copy.deepcopy(m)) for k, m in models.items()],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    ensemble_pred = ensemble.predict(X_test)
    predictions['Ensemble (Voting)'] = ensemble_pred
    scores['Ensemble (Voting)'] = accuracy_score(y_test, ensemble_pred)
    cv = cross_val_score(ensemble, X_train, y_train, cv=5)
    cv_means['Ensemble (Voting)'] = cv.mean()
    cv_stds['Ensemble (Voting)'] = cv.std()

    # Compile Results
    metrics_list = []
    for model_name in predictions:
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, predictions[model_name], average='weighted')
        metrics_list.append({
            'Model': model_name,
            'Accuracy': round(scores[model_name], 4),
            'Precision': round(precision, 4),
            'Recall': round(recall, 4),
            'F1-Score': round(f1, 4),
            'CV_Mean': round(cv_means[model_name], 4),
            'CV_Std': round(cv_stds[model_name], 4)
        })

    df = pd.DataFrame(metrics_list)
    print(f"\n📊 Results for {method_name}")
    print(df.to_string(index=False))
    df.to_csv(f'results_{method_name.lower().replace(" ", "_")}.csv', index=False)
    return df


# 🔠 TF-IDF
tfidf_vectorizer = TfidfVectorizer(
    stop_words='english', max_features=5000, min_df=2, max_df=0.8, ngram_range=(1, 2)
)
X_tfidf = tfidf_vectorizer.fit_transform(filtered_texts)
results_tfidf = run_experiment(X_tfidf, y, "TF-IDF")

# 🔠 Count Vectorizer
count_vectorizer = CountVectorizer(
    stop_words='english', max_features=5000, min_df=2, max_df=0.8, ngram_range=(1, 2)
)
X_count = count_vectorizer.fit_transform(filtered_texts)
results_count = run_experiment(X_count, y, "Count Vectorizer")

# 🔠 Hashing Vectorizer
hash_vectorizer = HashingVectorizer(
    stop_words='english', n_features=5000, alternate_sign=False, ngram_range=(1, 2)
)
X_hash = hash_vectorizer.transform(filtered_texts)
results_hash = run_experiment(X_hash, y, "Hashing Vectorizer")

# 📋 Combine all results
results_tfidf['Feature'] = 'TF-IDF'
results_count['Feature'] = 'Count Vectorizer'
results_hash['Feature'] = 'Hashing Vectorizer'

final_df = pd.concat([results_tfidf, results_count, results_hash], ignore_index=True)
final_df.to_csv('all_results_combined.csv', index=False)

print("\n✅ All feature extraction experiments completed successfully!")