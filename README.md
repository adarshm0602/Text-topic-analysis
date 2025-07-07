# Text-topic-analysis
using NLP and ML algorithms.

# 📊 Unsupervised Text Clustering Report

**Total documents used:** 59

**Vectorization:** TF-IDF (max 1000 features, bigrams, min_df=2, max_df=0.8)

**Optimal number of clusters:** 10


## ✅ K-Means

- **Silhouette Score:** 0.33

- **Adjusted Rand Index:** 0.06

- **Number of Clusters:** 10


## ✅ Hierarchical

- **Silhouette Score:** 0.33

- **Adjusted Rand Index:** 0.08

- **Number of Clusters:** 10


## ✅ DBSCAN

- **Silhouette Score:** 0.06

- **Adjusted Rand Index:** 0.05

- **Number of Clusters:** 2


## ✅ Gaussian Mixture

- **Silhouette Score:** 0.30

- **Adjusted Rand Index:** 0.10

- **Number of Clusters:** 10


## ✅ LDA

- **Silhouette Score:** 0.76

- **Adjusted Rand Index:** 0.04

- **Number of Clusters:** 10


🏆 **Best Performing Algorithm:** LDA with a Silhouette Score of 0.76






# 📊 Supervised Model Comparison Report

**Total documents used:** 58

**TF-IDF Features:** 45

**Number of unique labels:** 7


## 📋 Model Performance Metrics

| Model                  |   Accuracy |   Precision |   Recall |   F1-Score |   CV_Mean |   CV_Std |
|:-----------------------|-----------:|------------:|---------:|-----------:|----------:|---------:|
| K-Nearest Neighbors    |     0.3333 |      0.287  |   0.3333 |     0.2551 |    0.2778 |   0.1315 |
| Random Forest          |     0.6667 |      0.7222 |   0.6667 |     0.6389 |    0.52   |   0.1181 |
| Support Vector Machine |     0.75   |      0.8333 |   0.75   |     0.75   |    0.5422 |   0.0518 |
| Logistic Regression    |     0.75   |      0.9    |   0.75   |     0.7619 |    0.4978 |   0.097  |
| Naive Bayes            |     0.75   |      0.8333 |   0.75   |     0.75   |    0.4978 |   0.1556 |
| Ensemble (Voting)      |     0.5833 |      0.7143 |   0.5833 |     0.5741 |    0.54   |   0.1369 |


## 🏆 Model Ranking (by Accuracy)

1. **Support Vector Machine** — Accuracy: **0.7500** (🥇 Excellent)
2. **Logistic Regression** — Accuracy: **0.7500** (🥈 Very Good)
3. **Naive Bayes** — Accuracy: **0.7500** (🥉 Good)
4. **Random Forest** — Accuracy: **0.6667** (👍 Fair)
5. **Ensemble (Voting)** — Accuracy: **0.5833** (📊 Average)
6. **K-Nearest Neighbors** — Accuracy: **0.3333** (⚡ Baseline)


## 📈 Summary Statistics

- **Mean Accuracy:** 0.6389
- **Std Accuracy:** 0.1639
- **Max Accuracy:** 0.7500
- **Min Accuracy:** 0.3333
- **Accuracy Range:** 0.4167

## 🥇 Best Model Summary

- **Best Model:** Support Vector Machine
- **Test Accuracy:** 0.7500
- **CV Mean Accuracy:** 0.5422
- **Precision:** 0.8333
- **Recall:** 0.7500
- **F1-Score:** 0.7500

## 📋 Detailed Classification Report: Support Vector Machine

```

                 precision    recall  f1-score   support

    Environment       0.50      0.50      0.50         2
           Jobs       1.00      1.00      1.00         2
         Movies       1.00      1.00      1.00         2
         Sports       1.00      0.50      0.67         2
general message       1.00      0.50      0.67         2
 random message       0.50      1.00      0.67         2

       accuracy                           0.75        12
      macro avg       0.83      0.75      0.75        12
   weighted avg       0.83      0.75      0.75        12

```

