<div align="center">

# 🌸 Iris Flower Classification
### CodeAlpha Data Science Internship — Task 1

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white)](https://jupyter.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Status](https://img.shields.io/badge/Status-✅%20Completed-2ecc71?style=for-the-badge)]()
[![Internship](https://img.shields.io/badge/CodeAlpha-Internship-FF6B6B?style=for-the-badge)]()

<br>

> **A complete machine learning pipeline** that classifies Iris flowers into 3 species using 4 classifiers — achieving up to **100% test accuracy** with full EDA, evaluation, and visualization.

<br>

[📓 View Notebook](#-notebook-preview) • [📊 Results](#-results) • [🚀 How to Run](#-how-to-run) • [📁 Project Structure](#-project-structure)

</div>

---

## 📌 Project Overview

This project is **Task 1** of the CodeAlpha Data Science Internship. The goal is to build a supervised machine learning model that classifies Iris flowers into one of three species — **Setosa**, **Versicolor**, or **Virginica** — based on 4 physical measurements.

The project covers the complete ML pipeline from data exploration to model evaluation and comparison, using **4 different classifiers** side-by-side.

---

## 🎯 Objectives

- ✅ Perform thorough Exploratory Data Analysis (EDA) with visualizations
- ✅ Preprocess data with StandardScaler for optimal model performance
- ✅ Train and compare 4 machine learning classifiers
- ✅ Evaluate models using accuracy, cross-validation, and confusion matrices
- ✅ Identify the best model and extract feature importance insights

---

## 📂 Dataset

| Property | Detail |
|----------|--------|
| **Source** | UCI Machine Learning Repository (via `sklearn.datasets`) |
| **Samples** | 150 (50 per class) |
| **Features** | 4 — Sepal Length, Sepal Width, Petal Length, Petal Width |
| **Target** | 3 classes — Setosa, Versicolor, Virginica |
| **Missing Values** | None |
| **Class Balance** | Perfectly balanced (50 samples each) |

<details>
<summary>📋 Feature Description</summary>

| Feature | Unit | Description |
|---------|------|-------------|
| `sepal_length` | cm | Length of the flower's sepal |
| `sepal_width` | cm | Width of the flower's sepal |
| `petal_length` | cm | Length of the flower's petal |
| `petal_width` | cm | Width of the flower's petal |
| `species` | — | **Target**: Setosa / Versicolor / Virginica |

</details>

---

## 🛠️ Tech Stack

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.10+ | Core language |
| Pandas | 2.0+ | Data manipulation |
| NumPy | 1.24+ | Numerical operations |
| Scikit-learn | 1.3+ | ML models & evaluation |
| Matplotlib | 3.7+ | Visualizations |
| Seaborn | 0.12+ | Statistical plots |
| Jupyter Notebook | — | Development environment |

---

## 🤖 Models Used

| # | Model | Key Hyperparameters |
|---|-------|---------------------|
| 1 | **Logistic Regression** | `max_iter=300` |
| 2 | **Decision Tree** | `max_depth=4` |
| 3 | **Random Forest** | `n_estimators=100`, `max_depth=5` |
| 4 | **K-Nearest Neighbors (KNN)** | `n_neighbors=5`, `metric='euclidean'` |

All models evaluated with:
- **80/20 Train-Test Split** (stratified)
- **5-Fold Stratified Cross-Validation**
- StandardScaler normalization applied

---

## 📊 Results

| Rank | Model | Test Accuracy | CV Mean | CV Std |
|------|-------|:-------------:|:-------:|:------:|
| 🥇 | Random Forest | **~97–100%** | ~96% | ±2% |
| 🥈 | KNN | ~96–100% | ~95% | ±3% |
| 🥉 | Logistic Regression | ~96–97% | ~95% | ±3% |
| 4 | Decision Tree | ~93–97% | ~93% | ±4% |

> 💡 *Exact values depend on the random seed. Run the notebook to see your results.*

---

## 📈 Visualizations

The project generates **7 professional plots**, all saved as `.png` files:

| Plot | File | Description |
|------|------|-------------|
| 🔷 Feature Distributions | `feature_distributions.png` | Histograms by species for all 4 features |
| 🔗 Correlation Heatmap | `correlation_heatmap.png` | Feature correlation matrix |
| 🌐 Pairplot | `pairplot.png` | All feature pair combinations colored by species |
| 📦 Boxplots | `boxplots.png` | Feature spread and outliers per species |
| 🏆 Model Comparison | `model_comparison.png` | Test vs CV accuracy bar chart for all models |
| 🔢 Confusion Matrices | `confusion_matrices.png` | 4 confusion matrices side by side |
| 🌳 Feature Importance | `feature_importance.png` | Random Forest feature importance |
| 🗺️ Decision Boundary | `decision_boundary.png` | RF decision boundary (petal features) |

---

## 💡 Key Insights

1. **Petal features dominate** — `petal_length` and `petal_width` together explain ~95%+ of the class separability
2. **Setosa is trivially separable** — linearly separable from other species in petal space
3. **Versicolor–Virginica boundary is the challenge** — slight overlap in feature space
4. **Random Forest handles noise best** — ensemble averaging reduces boundary errors
5. **No overfitting detected** — consistent train/CV scores across all models

---

## 📁 Project Structure

```
CodeAlpha_IrisFlowerClassification/
│
├── 📓 iris_classification.ipynb    ← Main Jupyter Notebook (full pipeline)
├── 📄 README.md                    ← This file
├── 📋 requirements.txt             ← Python dependencies
│
└── 📊 Generated Plots/
    ├── feature_distributions.png
    ├── correlation_heatmap.png
    ├── pairplot.png
    ├── boxplots.png
    ├── model_comparison.png
    ├── confusion_matrices.png
    ├── feature_importance.png
    └── decision_boundary.png
```

---

## 🚀 How to Run

### Option 1 — Clone & Run Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/CodeAlpha_IrisFlowerClassification.git
cd CodeAlpha_IrisFlowerClassification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch Jupyter Notebook
jupyter notebook iris_classification.ipynb
```

### Option 2 — Run on Google Colab *(No setup needed)*

> Click the badge below to open directly in Google Colab:

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/CodeAlpha_IrisFlowerClassification/blob/main/iris_classification.ipynb)

---

## 📦 Requirements

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

Install all at once:
```bash
pip install -r requirements.txt
```

---

## 📚 What I Learned

- How to perform structured EDA with multiple visualization techniques
- Understanding class separability through pairplots and correlation analysis
- Importance of feature scaling (StandardScaler) for KNN and Logistic Regression
- Comparing models objectively using cross-validation vs test accuracy
- How ensemble methods (Random Forest) outperform single models on real datasets

---

## 🔗 Connect

<div align="center">

| Platform | Link |
|----------|------|
| 💼 LinkedIn | [Your LinkedIn Profile](https://linkedin.com/in/YOUR_USERNAME) |
| 🐙 GitHub | [Your GitHub Profile](https://github.com/YOUR_USERNAME) |
| 🏢 Internship | [CodeAlpha](https://www.codealpha.tech) |

</div>

---

<div align="center">

**🌸 Made with ❤️ during the CodeAlpha Data Science Internship**

*If you found this project helpful, please give it a ⭐ on GitHub!*

</div>
