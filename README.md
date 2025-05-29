# 🧠 Turkish Tweet Emotion Classification

This project performs emotion classification on Turkish tweets using multiple machine learning models. It includes both classical models (SVM, Random Forest, Gradient Boosting) and a lightweight neural model based on BERT embeddings.

---

## 📌 Project Overview

- **Goal**: Classify Turkish tweets into emotional categories.
- **Language**: Turkish
- **Techniques Used**:
  - Text preprocessing and vectorization (TF-IDF)
  - Embedding generation using BERT
  - Supervised learning models for classification
  - Evaluation with precision, recall, F1-score, and accuracy

---

## 📂 Dataset

- **Name**: [Turkish Tweet Emotion Dataset](https://www.kaggle.com/datasets/anil1055/turkish-tweet-dataset)
- **Source**: Kaggle
- **Download**: [https://www.kaggle.com/datasets/anil1055/turkish-tweet-dataset](https://www.kaggle.com/datasets/anil1055/turkish-tweet-dataset)

This dataset contains Turkish tweets labeled with one of five emotional categories. It is used to train and evaluate emotion classification models.

### 📋 Format

| Column     | Description                          |
|------------|--------------------------------------|
| `Tweet`    | The content of the tweet (in Turkish)|
| `Etiket`   | The emotion label                    |

### 🎯 Emotion Classes

- `kızgın` → anger  
- `korku` → fear  
- `mutlu` → happy  
- `surpriz` → surprise  
- `üzgün` → sad

These labels are encoded using `LabelEncoder` before training and used for supervised classification.

---

## 🛠️ Technologies & Libraries

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `scikit-learn`:
  - `SVC` – Support Vector Machine
  - `RandomForestClassifier` – Random Forest
  - `GradientBoostingClassifier` – Gradient Boosting
  - `TfidfVectorizer`, `LabelEncoder`, evaluation metrics
- `tensorflow`, `transformers` for BERT embeddings

---

## 🧪 Models Used

| Model                    | Description                      |
|--------------------------|----------------------------------|
| Support Vector Machine   | Classical linear classifier      |
| Random Forest            | Ensemble tree-based model        |
| Gradient Boosting        | Boosted decision trees           |
| BERT + Dense Layer       | Lightweight neural classifier using BERT embeddings |

---

## 📈 Evaluation Metrics

Each model is evaluated using:
- **Accuracy**
- **Precision**, **Recall**, **F1-Score**
- **Confusion Matrix**

Visualizations include:
- Classification report heatmaps
- Accuracy comparison bar plots
- Confusion matrices per model
