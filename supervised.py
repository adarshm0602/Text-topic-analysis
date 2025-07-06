import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import all supervised learning models
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

# For ensemble methods
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import precision_recall_fscore_support

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

# 📊 Filter out labels with < 2 samples
label_counts = Counter(labels)
filtered_texts, filtered_labels = [], []
for text, label in zip(texts, labels):
    if label_counts[label] >= 2:
        filtered_texts.append(text)
        filtered_labels.append(label)

print(f"📊 After filtering: {len(filtered_texts)} documents")
print(f"📊 Filtered label distribution: {Counter(filtered_labels)}")

# 📝 Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2)
)
X = vectorizer.fit_transform(filtered_texts)
y = filtered_labels

print(f"📊 TF-IDF matrix shape: {X.shape}")
print(f"📊 Number of unique labels: {len(set(y))}")

# 📊 Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"📊 Training set size: {X_train.shape[0]}")
print(f"📊 Test set size: {X_test.shape[0]}")

# 🤖 DEFINE ALL MODELS
print("\n" + "="*60)
print("🤖 TRAINING 5 SUPERVISED LEARNING MODELS")
print("="*60)

# Model 1: K-Nearest Neighbors
print("\n1️⃣ Training K-Nearest Neighbors...")
knn = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)
knn_cv_scores = cross_val_score(knn, X_train, y_train, cv=5)
print(f"✅ KNN Accuracy: {knn_accuracy:.4f}")

# Model 2: Random Forest
print("\n2️⃣ Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)
rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
print(f"✅ Random Forest Accuracy: {rf_accuracy:.4f}")

# Model 3: Support Vector Machine
print("\n3️⃣ Training Support Vector Machine...")
svm = SVC(kernel='rbf', random_state=42, probability=True)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_pred)
svm_cv_scores = cross_val_score(svm, X_train, y_train, cv=5)
print(f"✅ SVM Accuracy: {svm_accuracy:.4f}")

# Model 4: Logistic Regression
print("\n4️⃣ Training Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_accuracy = accuracy_score(y_test, lr_pred)
lr_cv_scores = cross_val_score(lr, X_train, y_train, cv=5)
print(f"✅ Logistic Regression Accuracy: {lr_accuracy:.4f}")

# Model 5: Naive Bayes
print("\n5️⃣ Training Naive Bayes...")
nb = MultinomialNB(alpha=1.0)
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)
nb_cv_scores = cross_val_score(nb, X_train, y_train, cv=5)
print(f"✅ Naive Bayes Accuracy: {nb_accuracy:.4f}")

# 🏆 ENSEMBLE METHOD - VOTING CLASSIFIER
print("\n" + "="*60)
print("🏆 CREATING ENSEMBLE MODEL")
print("="*60)

# Create ensemble with all models
ensemble = VotingClassifier(
    estimators=[
        ('knn', knn),
        ('rf', rf),
        ('svm', svm),
        ('lr', lr),
        ('nb', nb)
    ],
    voting='soft'  # Use probability voting
)

ensemble.fit(X_train, y_train)
ensemble_pred = ensemble.predict(X_test)
ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
ensemble_cv_scores = cross_val_score(ensemble, X_train, y_train, cv=5)
print(f"✅ Ensemble Accuracy: {ensemble_accuracy:.4f}")

# 📊 CALCULATE DETAILED METRICS FOR ALL MODELS
def calculate_metrics(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    return {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

# Calculate metrics for all models
all_metrics = []
models_data = [
    (knn_pred, 'K-Nearest Neighbors', knn_cv_scores),
    (rf_pred, 'Random Forest', rf_cv_scores),
    (svm_pred, 'Support Vector Machine', svm_cv_scores),
    (lr_pred, 'Logistic Regression', lr_cv_scores),
    (nb_pred, 'Naive Bayes', nb_cv_scores),
    (ensemble_pred, 'Ensemble (Voting)', ensemble_cv_scores)
]

for pred, name, cv_scores in models_data:
    metrics = calculate_metrics(y_test, pred, name)
    metrics['CV_Mean'] = cv_scores.mean()
    metrics['CV_Std'] = cv_scores.std()
    all_metrics.append(metrics)

# 📋 CREATE COMPREHENSIVE RESULTS TABLE
print("\n" + "="*80)
print("📋 COMPREHENSIVE RESULTS TABLE - TABLEAU FORMAT")
print("="*80)

results_df = pd.DataFrame(all_metrics)
results_df['Accuracy'] = results_df['Accuracy'].round(4)
results_df['Precision'] = results_df['Precision'].round(4)
results_df['Recall'] = results_df['Recall'].round(4)
results_df['F1-Score'] = results_df['F1-Score'].round(4)
results_df['CV_Mean'] = results_df['CV_Mean'].round(4)
results_df['CV_Std'] = results_df['CV_Std'].round(4)

# Create a more detailed tableau-style table
print("\n🎯 MAIN PERFORMANCE METRICS TABLE")
print("┌" + "─" * 28 + "┬" + "─" * 12 + "┬" + "─" * 12 + "┬" + "─" * 10 + "┬" + "─" * 10 + "┐")
print("│{:<28}│{:>12}│{:>12}│{:>10}│{:>10}│".format("Model", "Accuracy", "Precision", "Recall", "F1-Score"))
print("├" + "─" * 28 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 10 + "┼" + "─" * 10 + "┤")

for _, row in results_df.iterrows():
    print("│{:<28}│{:>12.4f}│{:>12.4f}│{:>10.4f}│{:>10.4f}│".format(
        row['Model'], row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score']))

print("└" + "─" * 28 + "┴" + "─" * 12 + "┴" + "─" * 12 + "┴" + "─" * 10 + "┴" + "─" * 10 + "┘")

print("\n📊 CROSS-VALIDATION RESULTS TABLE")
print("┌" + "─" * 28 + "┬" + "─" * 15 + "┬" + "─" * 15 + "┐")
print("│{:<28}│{:>15}│{:>15}│".format("Model", "CV Mean", "CV Std Dev"))
print("├" + "─" * 28 + "┼" + "─" * 15 + "┼" + "─" * 15 + "┤")

for _, row in results_df.iterrows():
    print("│{:<28}│{:>15.4f}│{:>15.4f}│".format(
        row['Model'], row['CV_Mean'], row['CV_Std']))

print("└" + "─" * 28 + "┴" + "─" * 15 + "┴" + "─" * 15 + "┘")

print("\n🏆 RANKING TABLE (By Accuracy)")
print("┌" + "─" * 6 + "┬" + "─" * 28 + "┬" + "─" * 12 + "┬" + "─" * 15 + "┐")
print("│{:<6}│{:<28}│{:>12}│{:>15}│".format("Rank", "Model", "Accuracy", "Performance"))
print("├" + "─" * 6 + "┼" + "─" * 28 + "┼" + "─" * 12 + "┼" + "─" * 15 + "┤")

# Sort by accuracy for ranking
sorted_results = results_df.sort_values('Accuracy', ascending=False).reset_index(drop=True)
performance_levels = ['🥇 Excellent', '🥈 Very Good', '🥉 Good', '👍 Fair', '📊 Average', '⚡ Baseline']

for i, (_, row) in enumerate(sorted_results.iterrows()):
    perf = performance_levels[i] if i < len(performance_levels) else '📊 Average'
    print("│{:<6}│{:<28}│{:>12.4f}│{:>15}│".format(
        i+1, row['Model'], row['Accuracy'], perf))

print("└" + "─" * 6 + "┴" + "─" * 28 + "┴" + "─" * 12 + "┴" + "─" * 15 + "┘")

# Additional summary statistics
print("\n📈 SUMMARY STATISTICS")
print("┌" + "─" * 20 + "┬" + "─" * 15 + "┐")
print("│{:<20}│{:>15}│".format("Statistic", "Value"))
print("├" + "─" * 20 + "┼" + "─" * 15 + "┤")
print("│{:<20}│{:>15.4f}│".format("Mean Accuracy", results_df['Accuracy'].mean()))
print("│{:<20}│{:>15.4f}│".format("Std Accuracy", results_df['Accuracy'].std()))
print("│{:<20}│{:>15.4f}│".format("Max Accuracy", results_df['Accuracy'].max()))
print("│{:<20}│{:>15.4f}│".format("Min Accuracy", results_df['Accuracy'].min()))
print("│{:<20}│{:>15.4f}│".format("Accuracy Range", results_df['Accuracy'].max() - results_df['Accuracy'].min()))
print("└" + "─" * 20 + "┴" + "─" * 15 + "┘")

print(results_df.to_string(index=False))

# 📊 ACCURACY COMPARISON CHART
print("\n" + "="*60)
print("📊 CREATING VISUALIZATION")
print("="*60)

# Create accuracy comparison chart
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Chart 1: Test Accuracy Comparison
models = results_df['Model']
accuracies = results_df['Accuracy']
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3']

bars1 = ax1.bar(models, accuracies, color=colors)
ax1.set_title('Test Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy')
ax1.set_ylim(0, 1)
ax1.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, acc in zip(bars1, accuracies):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
             f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')

# Chart 2: Cross-Validation Scores
cv_means = results_df['CV_Mean']
cv_stds = results_df['CV_Std']

bars2 = ax2.bar(models, cv_means, yerr=cv_stds, color=colors, alpha=0.7, capsize=5)
ax2.set_title('Cross-Validation Accuracy (5-Fold)', fontsize=14, fontweight='bold')
ax2.set_ylabel('CV Accuracy')
ax2.set_ylim(0, 1)
ax2.tick_params(axis='x', rotation=45)

# Add value labels on bars
for bar, mean, std in zip(bars2, cv_means, cv_stds):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
             f'{mean:.3f}±{std:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.show()

# 📊 DETAILED METRICS HEATMAP
fig, ax = plt.subplots(figsize=(10, 6))
metrics_for_heatmap = results_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score']].set_index('Model')
sns.heatmap(metrics_for_heatmap, annot=True, cmap='YlOrRd', fmt='.3f', ax=ax)
ax.set_title('Model Performance Metrics Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# 🏆 IDENTIFY BEST MODEL
best_model_idx = results_df['Accuracy'].idxmax()
best_model = results_df.loc[best_model_idx]

print("\n" + "="*60)
print("🏆 BEST MODEL SUMMARY")
print("="*60)
print(f"🥇 Best Model: {best_model['Model']}")
print(f"📊 Test Accuracy: {best_model['Accuracy']:.4f}")
print(f"📊 Cross-Validation: {best_model['CV_Mean']:.4f} ± {best_model['CV_Std']:.4f}")
print(f"📊 Precision: {best_model['Precision']:.4f}")
print(f"📊 Recall: {best_model['Recall']:.4f}")
print(f"📊 F1-Score: {best_model['F1-Score']:.4f}")

# 📋 DETAILED CLASSIFICATION REPORT FOR BEST MODEL
print(f"\n📋 Detailed Classification Report for {best_model['Model']}:")
print("-" * 60)

# Get predictions from best model
if best_model['Model'] == 'K-Nearest Neighbors':
    best_pred = knn_pred
elif best_model['Model'] == 'Random Forest':
    best_pred = rf_pred
elif best_model['Model'] == 'Support Vector Machine':
    best_pred = svm_pred
elif best_model['Model'] == 'Logistic Regression':
    best_pred = lr_pred
elif best_model['Model'] == 'Naive Bayes':
    best_pred = nb_pred
else:
    best_pred = ensemble_pred

print(classification_report(y_test, best_pred))

# 📊 CONFUSION MATRIX FOR BEST MODEL
fig, ax = plt.subplots(figsize=(8, 6))
cm = confusion_matrix(y_test, best_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_title(f'Confusion Matrix - {best_model["Model"]}', fontsize=14, fontweight='bold')
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
plt.tight_layout()
plt.show()

# 💾 SAVE RESULTS
results_df.to_csv('model_comparison_results.csv', index=False)
print(f"\n💾 Results saved to 'model_comparison_results.csv'")

print("\n✅ Analysis Complete!")
print(f"🎯 Best performing model: {best_model['Model']} with {best_model['Accuracy']:.4f} accuracy")



# 📊 SAVE RESULTS TO MARKDOWN FILE
# 📑 SAVE REPORT TO MARKDOWN FILE
report_lines = []
report_lines.append("# 📊 Supervised Model Comparison Report\n")
report_lines.append(f"**Total documents used:** {len(filtered_texts)}\n")
report_lines.append(f"**TF-IDF Features:** {X.shape[1]}\n")
report_lines.append(f"**Number of unique labels:** {len(set(y))}\n\n")

# 📋 Add metrics table
report_lines.append("## 📋 Model Performance Metrics\n")
report_lines.append(results_df.to_markdown(index=False))
report_lines.append("\n")

# 📊 Ranking Table
report_lines.append("## 🏆 Model Ranking (by Accuracy)\n")
for i, (_, row) in enumerate(sorted_results.iterrows()):
    perf = performance_levels[i] if i < len(performance_levels) else '📊 Average'
    report_lines.append(f"{i+1}. **{row['Model']}** — Accuracy: **{row['Accuracy']:.4f}** ({perf})")

report_lines.append("\n")

# 📊 Summary Statistics
report_lines.append("## 📈 Summary Statistics\n")
report_lines.append(f"- **Mean Accuracy:** {results_df['Accuracy'].mean():.4f}")
report_lines.append(f"- **Std Accuracy:** {results_df['Accuracy'].std():.4f}")
report_lines.append(f"- **Max Accuracy:** {results_df['Accuracy'].max():.4f}")
report_lines.append(f"- **Min Accuracy:** {results_df['Accuracy'].min():.4f}")
report_lines.append(f"- **Accuracy Range:** {(results_df['Accuracy'].max() - results_df['Accuracy'].min()):.4f}\n")

# 📊 Best Model Summary
report_lines.append("## 🥇 Best Model Summary\n")
report_lines.append(f"- **Best Model:** {best_model['Model']}")
report_lines.append(f"- **Test Accuracy:** {best_model['Accuracy']:.4f}")
report_lines.append(f"- **CV Mean Accuracy:** {best_model['CV_Mean']:.4f}")
report_lines.append(f"- **Precision:** {best_model['Precision']:.4f}")
report_lines.append(f"- **Recall:** {best_model['Recall']:.4f}")
report_lines.append(f"- **F1-Score:** {best_model['F1-Score']:.4f}\n")

# 📋 Classification Report for Best Model
report_lines.append(f"## 📋 Detailed Classification Report: {best_model['Model']}\n")
report_lines.append("```\n")
report_lines.append(classification_report(y_test, best_pred))
report_lines.append("```\n")


# 📑 SAVE REPORT TO MARKDOWN FILE
report_lines = []
report_lines.append("# 📊 Supervised Model Comparison Report\n")
report_lines.append(f"**Total documents used:** {len(filtered_texts)}\n")
report_lines.append(f"**TF-IDF Features:** {X.shape[1]}\n")
report_lines.append(f"**Number of unique labels:** {len(set(y))}\n\n")

# 📋 Add metrics table
report_lines.append("## 📋 Model Performance Metrics\n")
report_lines.append(results_df.to_markdown(index=False))
report_lines.append("\n")

# 📊 Ranking Table
report_lines.append("## 🏆 Model Ranking (by Accuracy)\n")
for i, (_, row) in enumerate(sorted_results.iterrows()):
    perf = performance_levels[i] if i < len(performance_levels) else '📊 Average'
    report_lines.append(f"{i+1}. **{row['Model']}** — Accuracy: **{row['Accuracy']:.4f}** ({perf})")

report_lines.append("\n")

# 📊 Summary Statistics
report_lines.append("## 📈 Summary Statistics\n")
report_lines.append(f"- **Mean Accuracy:** {results_df['Accuracy'].mean():.4f}")
report_lines.append(f"- **Std Accuracy:** {results_df['Accuracy'].std():.4f}")
report_lines.append(f"- **Max Accuracy:** {results_df['Accuracy'].max():.4f}")
report_lines.append(f"- **Min Accuracy:** {results_df['Accuracy'].min():.4f}")
report_lines.append(f"- **Accuracy Range:** {(results_df['Accuracy'].max() - results_df['Accuracy'].min()):.4f}\n")

# 📊 Best Model Summary
report_lines.append("## 🥇 Best Model Summary\n")
report_lines.append(f"- **Best Model:** {best_model['Model']}")
report_lines.append(f"- **Test Accuracy:** {best_model['Accuracy']:.4f}")
report_lines.append(f"- **CV Mean Accuracy:** {best_model['CV_Mean']:.4f}")
report_lines.append(f"- **Precision:** {best_model['Precision']:.4f}")
report_lines.append(f"- **Recall:** {best_model['Recall']:.4f}")
report_lines.append(f"- **F1-Score:** {best_model['F1-Score']:.4f}\n")

# 📋 Classification Report for Best Model
report_lines.append(f"## 📋 Detailed Classification Report: {best_model['Model']}\n")
report_lines.append("```\n")
report_lines.append(classification_report(y_test, best_pred))
report_lines.append("```\n")

# 📑 SAVE REPORT TO MARKDOWN FILE
report_lines = []
report_lines.append("# 📊 Supervised Model Comparison Report\n")
report_lines.append(f"**Total documents used:** {len(filtered_texts)}\n")
report_lines.append(f"**TF-IDF Features:** {X.shape[1]}\n")
report_lines.append(f"**Number of unique labels:** {len(set(y))}\n\n")

# 📋 Add metrics table
report_lines.append("## 📋 Model Performance Metrics\n")
report_lines.append(results_df.to_markdown(index=False))
report_lines.append("\n")

# 📊 Ranking Table
report_lines.append("## 🏆 Model Ranking (by Accuracy)\n")
for i, (_, row) in enumerate(sorted_results.iterrows()):
    perf = performance_levels[i] if i < len(performance_levels) else '📊 Average'
    report_lines.append(f"{i+1}. **{row['Model']}** — Accuracy: **{row['Accuracy']:.4f}** ({perf})")

report_lines.append("\n")

# 📊 Summary Statistics
report_lines.append("## 📈 Summary Statistics\n")
report_lines.append(f"- **Mean Accuracy:** {results_df['Accuracy'].mean():.4f}")
report_lines.append(f"- **Std Accuracy:** {results_df['Accuracy'].std():.4f}")
report_lines.append(f"- **Max Accuracy:** {results_df['Accuracy'].max():.4f}")
report_lines.append(f"- **Min Accuracy:** {results_df['Accuracy'].min():.4f}")
report_lines.append(f"- **Accuracy Range:** {(results_df['Accuracy'].max() - results_df['Accuracy'].min()):.4f}\n")

# 📊 Best Model Summary
report_lines.append("## 🥇 Best Model Summary\n")
report_lines.append(f"- **Best Model:** {best_model['Model']}")
report_lines.append(f"- **Test Accuracy:** {best_model['Accuracy']:.4f}")
report_lines.append(f"- **CV Mean Accuracy:** {best_model['CV_Mean']:.4f}")
report_lines.append(f"- **Precision:** {best_model['Precision']:.4f}")
report_lines.append(f"- **Recall:** {best_model['Recall']:.4f}")
report_lines.append(f"- **F1-Score:** {best_model['F1-Score']:.4f}\n")

# 📋 Classification Report for Best Model
report_lines.append(f"## 📋 Detailed Classification Report: {best_model['Model']}\n")
report_lines.append("```\n")
report_lines.append(classification_report(y_test, best_pred))
report_lines.append("```\n")

# Save to markdown file
with open("supervised_model_comparison.md", "w") as f:
    f.writelines([line + "\n" for line in report_lines])

print("\n✅ Markdown report saved to 'supervised_model_comparison.md'")


