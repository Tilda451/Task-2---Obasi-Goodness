# Task-2---Obasi-Goodness
# DecodeLabs — Project 2: Data Classification Using AI
**Industrial Training Kit | Batch 2026**

---

## Overview

This project implements a complete **supervised machine learning pipeline** using the
**K-Nearest Neighbors (KNN)** algorithm to classify iris flowers into three species:
*Setosa*, *Versicolor*, and *Virginica*.

It covers every stage of the IPO framework taught in the kit:

```
INPUT  →  Feature Scaling  →  PROCESS (KNN)  →  OUTPUT (Confusion Matrix + F1)
```

---

## Project Structure

```
project-2/
│
├── iris_classifier.py        # Main script (all steps in one file)
├── elbow_curve.png           # Auto-generated: K vs Error Rate plot
├── confusion_matrix.png      # Auto-generated: model diagnostic heatmap
├── feature_distributions.png # Auto-generated: per-feature class separation
└── README.md                 # This file
```

---

## Requirements

Install dependencies with:

```bash
pip install numpy scikit-learn matplotlib seaborn
```

| Library       | Purpose                              |
|---------------|--------------------------------------|
| `numpy`       | Numerical operations                 |
| `scikit-learn`| Dataset, model, metrics              |
| `matplotlib`  | Plotting charts                      |
| `seaborn`     | Confusion matrix heatmap             |

> Tested on Python 3.8+

---

## How to Run

```bash
python iris_classifier.py
```

That's it. The script will:
1. Load and explore the Iris dataset
2. Scale features with `StandardScaler`
3. Split data 80/20 (train/test) with shuffling
4. Find the optimal K using the Elbow Method (K=1 to 20)
5. Train the KNN model and make predictions
6. Print accuracy, F1 score, and a classification report
7. Save three chart images to your working directory
8. Run a live prediction demo on a new sample

---

## Concepts Applied

### 1. Feature Scaling
Raw features have different units (e.g., sepal length in cm vs. petal width).
KNN uses Euclidean distance — larger-scale features would dominate unfairly.
`StandardScaler` transforms each feature to **mean = 0, variance = 1**.

> The scaler is fit **only on training data** to prevent data leakage.

### 2. Train-Test Split (80/20)
- **Training set (80%)** — the model learns patterns from this
- **Test set (20%)** — locked away, used only to validate final performance
- `stratify=y` ensures each class is equally represented in both sets

### 3. K-Nearest Neighbors (KNN)
The **Proximity Principle**: similar things exist in close proximity.

For a new data point, KNN:
1. Calculates the distance to every training point
2. Selects the K nearest neighbors
3. Takes a **majority vote** → assigns that class label

```python
model = KNeighborsClassifier(n_neighbors=optimal_k)
model.fit(X_train, y_train)           # Memorise the map
predictions = model.predict(X_test)   # Apply logic
```

### 4. Choosing K — The Elbow Method
| K value       | Problem              |
|---------------|----------------------|
| K = 1         | Overfitting (noise)  |
| K = 100       | Underfitting (too generic) |
| K = optimal  | The "elbow" — lowest error rate |

The script automatically finds and uses the optimal K.

### 5. Output Validation

**Why not just use accuracy?**
On imbalanced datasets, a model predicting only the majority class can score 99% accuracy
while being completely useless. That's the *Accuracy Mirage*.

We use:

| Metric           | What it measures                                      |
|------------------|-------------------------------------------------------|
| Confusion Matrix | Full breakdown of TP, FP, FN, TN per class            |
| Precision        | Of all predicted positives, how many were correct?    |
| Recall           | Of all actual positives, how many did we catch?       |
| F1 Score         | Harmonic mean of Precision & Recall (balanced view)   |

---

## Sample Output

```
=======================================================
  DECODELABS — Project 2: Data Classification Using AI
=======================================================

Dataset Shape   : (150, 4)  (samples × features)
Classes         : ['setosa', 'versicolor', 'virginica']
Samples/class   : [50, 50, 50]  (balanced)

Train samples   : 120
Test  samples   : 30
Optimal K       : 5  (lowest error rate)

─────────────────────────────────────────────────────
  OUTPUT VALIDATION
─────────────────────────────────────────────────────
Accuracy Score  : 0.9667  (96.67%)
F1 Score (wtd)  : 0.9667

               precision  recall  f1-score  support
     setosa       1.00      1.00      1.00       10
 versicolor       1.00      0.90      0.95       10
  virginica       0.91      1.00      0.95       10

  LIVE PREDICTION DEMO
─────────────────────────────────────────────────────
Predicted class    : SETOSA
```

---

## Generated Charts

| File                        | Description                          |
|-----------------------------|--------------------------------------|
| `elbow_curve.png`           | K vs Error Rate — shows optimal K    |
| `confusion_matrix.png`      | Heatmap of TP/FP/FN/TN               |
| `feature_distributions.png` | How well each feature separates classes |

---

## Experimentation Ideas

You can try these using the same procedure:

- **Change the algorithm** — swap KNN for `DecisionTreeClassifier` or `SVC` and compare F1 scores
- **Change the split ratio** — try 70/30 or 90/10 and observe the effect on accuracy
- **Use a different dataset** — replace Iris with the Wine or Breast Cancer dataset from `sklearn.datasets`
- **Plot the decision boundary** — visualise how KNN separates classes in 2D (use only 2 features)
- **Cross-validation** — use `cross_val_score` for a more robust performance estimate

---


