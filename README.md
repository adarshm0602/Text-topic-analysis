# Text-topic-analysis
using NLP and ML algorithms.





# 📊 Text Classification Model Report

**Total documents used:** 58

**Train-Test Split:** 80-20 Stratified

## ✅ Naive Bayes
**Accuracy:** 66.67%

**Classification Report:**

```
                 precision    recall  f1-score   support

    Environment       1.00      0.50      0.67         2
           Jobs       1.00      1.00      1.00         2
         Movies       1.00      1.00      1.00         2
         Sports       0.00      0.00      0.00         2
general message       1.00      0.50      0.67         2
 random message       0.33      1.00      0.50         2

       accuracy                           0.67        12
      macro avg       0.72      0.67      0.64        12
   weighted avg       0.72      0.67      0.64        12

```

## ✅ Logistic Regression
**Accuracy:** 66.67%

**Classification Report:**

```
                 precision    recall  f1-score   support

    Environment       0.33      1.00      0.50         2
           Jobs       1.00      1.00      1.00         2
         Movies       1.00      1.00      1.00         2
         Sports       0.00      0.00      0.00         2
general message       1.00      0.50      0.67         2
 random message       1.00      0.50      0.67         2

       accuracy                           0.67        12
      macro avg       0.72      0.67      0.64        12
   weighted avg       0.72      0.67      0.64        12

```

## ✅ Linear SVM
**Accuracy:** 75.0%

**Classification Report:**

```
                 precision    recall  f1-score   support

    Environment       1.00      0.50      0.67         2
           Jobs       1.00      1.00      1.00         2
         Movies       1.00      1.00      1.00         2
         Sports       1.00      0.50      0.67         2
general message       1.00      0.50      0.67         2
 random message       0.40      1.00      0.57         2

       accuracy                           0.75        12
      macro avg       0.90      0.75      0.76        12
   weighted avg       0.90      0.75      0.76        12

```

## ✅ Random Forest
**Accuracy:** 41.67%

**Classification Report:**

```
                 precision    recall  f1-score   support

    Environment       0.00      0.00      0.00         2
           Jobs       1.00      1.00      1.00         2
         Movies       1.00      0.50      0.67         2
         Sports       0.00      0.00      0.00         2
general message       0.22      1.00      0.36         2
 random message       0.00      0.00      0.00         2

       accuracy                           0.42        12
      macro avg       0.37      0.42      0.34        12
   weighted avg       0.37      0.42      0.34        12

```

## ✅ K-Nearest Neighbors (k=3)
**Accuracy:** 41.67%

**Classification Report:**

```
                 precision    recall  f1-score   support

    Environment       0.25      0.50      0.33         2
           Jobs       0.50      1.00      0.67         2
         Movies       0.50      1.00      0.67         2
         Sports       0.00      0.00      0.00         2
general message       0.00      0.00      0.00         2
 random message       0.00      0.00      0.00         2

       accuracy                           0.42        12
      macro avg       0.21      0.42      0.28        12
   weighted avg       0.21      0.42      0.28        12

```
