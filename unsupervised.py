import os
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA, LatentDirichletAllocation
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, adjusted_rand_score
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# 📂 Paths
folder_path = 'posts_all/'
labels_file = 'labels.csv'

# 📥 Load labels (for evaluation purposes only)
df_labels = pd.read_csv(labels_file)

# 📖 Read texts
texts, true_labels = [], []
for _, row in df_labels.iterrows():
    file_path = os.path.join(folder_path, row['filename'])
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
            content = ''.join(lines[1:]).strip()  # Skip label line
            texts.append(content)
            true_labels.append(row['label'])

print(f"\U0001F4CA Loaded {len(texts)} documents")
print(f"\U0001F4CA True labels distribution: {Counter(true_labels)}")

# 📝 Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=1000,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2)
)
X = vectorizer.fit_transform(texts)
print(f"\U0001F4CA TF-IDF matrix shape: {X.shape}")

# 🎯 Determine optimal number of clusters using elbow method
def plot_elbow_method(X, max_clusters=10):
    inertias = []
    silhouette_scores = []
    K_range = range(2, max_clusters + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(K_range, inertias, 'bo-')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Inertia')
    ax1.set_title('Elbow Method for Optimal k')
    ax1.grid(True)

    ax2.plot(K_range, silhouette_scores, 'ro-')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title('Silhouette Score vs Number of Clusters')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()

    return K_range[np.argmax(silhouette_scores)]

# Find optimal number of clusters
optimal_k = plot_elbow_method(X.toarray())
print(f"\U0001F3AF Optimal number of clusters: {optimal_k}")

# 🤖 K-Means
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X)
kmeans_silhouette = silhouette_score(X, kmeans_labels)
kmeans_ari = adjusted_rand_score(true_labels, kmeans_labels)

# 🤖 Hierarchical
hierarchical = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
hierarchical_labels = hierarchical.fit_predict(X.toarray())
hierarchical_silhouette = silhouette_score(X, hierarchical_labels)
hierarchical_ari = adjusted_rand_score(true_labels, hierarchical_labels)

# 🤖 DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=3)
dbscan_labels = dbscan.fit_predict(X.toarray())
n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)
if n_clusters_dbscan > 1:
    dbscan_silhouette = silhouette_score(X, dbscan_labels)
    dbscan_ari = adjusted_rand_score(true_labels, dbscan_labels)
else:
    dbscan_silhouette, dbscan_ari = 0, 0

# 🤖 Gaussian Mixture
gmm = GaussianMixture(n_components=optimal_k, random_state=42)
gmm_labels = gmm.fit_predict(X.toarray())
gmm_silhouette = silhouette_score(X, gmm_labels)
gmm_ari = adjusted_rand_score(true_labels, gmm_labels)

# 🤖 LDA
count_vectorizer = CountVectorizer(stop_words='english', max_features=1000, min_df=2, max_df=0.8)
X_count = count_vectorizer.fit_transform(texts)
lda = LatentDirichletAllocation(n_components=optimal_k, random_state=42)
lda_topics = lda.fit_transform(X_count)
lda_labels = np.argmax(lda_topics, axis=1)
lda_silhouette = silhouette_score(lda_topics, lda_labels)
lda_ari = adjusted_rand_score(true_labels, lda_labels)

# 📊 Compare Results
results = pd.DataFrame({
    'Algorithm': ['K-Means', 'Hierarchical', 'DBSCAN', 'Gaussian Mixture', 'LDA'],
    'Silhouette Score': [kmeans_silhouette, hierarchical_silhouette, dbscan_silhouette, gmm_silhouette, lda_silhouette],
    'Adjusted Rand Index': [kmeans_ari, hierarchical_ari, dbscan_ari, gmm_ari, lda_ari],
    'Number of Clusters': [optimal_k, optimal_k, n_clusters_dbscan, optimal_k, optimal_k]
})

print(results.to_string(index=False))

# 📈 t-SNE Visualization
tsne_perplexity = min(30, len(texts) // 3)
# Fix: Use min of 50 or number of features to avoid PCA error
n_components = min(50, X.shape[1])
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X.toarray())
tsne = TSNE(n_components=2, random_state=42, perplexity=tsne_perplexity)
X_tsne = tsne.fit_transform(X_pca)

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('Cluster Visualization with t-SNE', fontsize=16)

algorithms = [('K-Means', kmeans_labels), ('Hierarchical', hierarchical_labels), ('Gaussian Mixture', gmm_labels), ('LDA', lda_labels)]

for idx, (name, labels) in enumerate(algorithms):
    ax = axes[idx // 2, idx % 2]
    scatter = ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.6)
    ax.set_title(f'{name} Clustering')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    plt.colorbar(scatter, ax=ax)

plt.tight_layout()
plt.show()

# 📝 Word Clouds (for best algorithm)
best_algorithm = results.loc[results['Silhouette Score'].idxmax(), 'Algorithm']
best_labels = kmeans_labels if best_algorithm == 'K-Means' else hierarchical_labels

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle(f'Word Clouds by Cluster ({best_algorithm})', fontsize=16)

for cluster_id in range(min(4, len(set(best_labels)))):
    cluster_texts = [texts[i] for i, label in enumerate(best_labels) if label == cluster_id]
    if not cluster_texts:
        continue
    cluster_text = ' '.join(cluster_texts)
    wordcloud = WordCloud(width=400, height=300, background_color='white').generate(cluster_text)
    ax = axes[cluster_id // 2, cluster_id % 2]
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.set_title(f'Cluster {cluster_id} ({len(cluster_texts)} documents)')
    ax.axis('off')

plt.tight_layout()
plt.show()

# 📑 Save Results to Markdown
report_lines = []
report_lines.append("# 📊 Unsupervised Text Clustering Report\n")
report_lines.append(f"**Total documents used:** {len(texts)}\n")
report_lines.append(f"**Vectorization:** TF-IDF (max 1000 features, bigrams, min_df=2, max_df=0.8)\n")
report_lines.append(f"**Optimal number of clusters:** {optimal_k}\n\n")

for _, row in results.iterrows():
    report_lines.append(f"## ✅ {row['Algorithm']}\n")
    report_lines.append(f"- **Silhouette Score:** {row['Silhouette Score']:.2f}\n")
    report_lines.append(f"- **Adjusted Rand Index:** {row['Adjusted Rand Index']:.2f}\n")
    report_lines.append(f"- **Number of Clusters:** {int(row['Number of Clusters'])}\n\n")

report_lines.append(f"🏆 **Best Performing Algorithm:** {best_algorithm} with a Silhouette Score of {results['Silhouette Score'].max():.2f}\n")

with open("unsupervised_report.md", "w") as f:
    f.writelines([line + "\n" for line in report_lines])

print("\n✅ Markdown report saved to 'unsupervised_report.md'")
