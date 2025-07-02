# 🎬 IMDb Movie Review Sentiment Analysis

## 📌 Overview

Sentiment analysis is a core task in Natural Language Processing (NLP), where the goal is to determine the sentiment expressed in a piece of text—whether it's positive or negative. In this project, we analyze **IMDb movie reviews** to predict sentiment based solely on review text.

By leveraging **text preprocessing techniques**, **feature extraction methods**, and various **classification algorithms**, we aim to develop an efficient machine learning model that can **accurately classify movie reviews**. This analysis can help content creators, streaming platforms, and studios understand audience perception and tailor strategies accordingly.

---

## 🎯 Problem Statement

The objective of this project is to:
- Build a **machine learning classification model** to predict whether a movie review is **positive or negative**.
- Apply **NLP preprocessing techniques** and **vectorization** to convert text data into usable features.
- Use various classification algorithms and evaluate their performance using appropriate metrics.

---

## 🗂️ Dataset Information

- **Dataset Name**: `IMDb` [Download dataset](https://docs.google.com/spreadsheets/d/106x15uz8ccQ6Wvpc8-sYjXisBN8vewS435I7z3wd4sw/edit?usp=sharing)
- **Features**:
  - `review`: The full text of the movie review.
  - `sentiment`: The sentiment label (either `positive` or `negative`).

---

## ✅ Deliverables

### 1. 🔍 Data Exploration & Preprocessing
- Explore the dataset for:
  - Missing values
  - Class imbalance
  - Review length distribution
- Clean the text:
  - Remove punctuation, HTML tags, special characters
  - Tokenization, lemmatization/stemming
  - Lowercasing and stopword removal
- Vectorize text using:
  - **Bag-of-Words**
  - **TF-IDF**

### 2. 🧠 Feature Engineering
- Extract numerical features from text:
  - TF-IDF features
  - Word count, character count, average word length
- Optional: use **Word2Vec**, **GloVe**, or pre-trained embeddings for deep learning models

### 3. ⚙️ Model Development
- Train and compare multiple models:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Random Forest
  - LSTM or BERT (if applicable)
- Use cross-validation and grid search for hyperparameter tuning

### 4. 📏 Model Evaluation
- Evaluate using:
  - **Accuracy**
  - **Precision**
  - **Recall**
  - **F1-score**
  - **Confusion Matrix**
  - **ROC-AUC Curve**

### 5. 📽️ Final Report & Presentation
- Prepare a final summary report documenting:
  - Exploratory data analysis
  - Preprocessing pipeline
  - Model performance
  - Key takeaways
---

## 🛠️ Tools & Technologies

- **Languages**: Python
- **Libraries**:
  - `pandas`, `numpy` – Data manipulation
  - `nltk`, `spaCy` – Text preprocessing
  - `scikit-learn` – Machine learning
  - `matplotlib`, `seaborn` – Visualization
  - `TensorFlow/Keras` – For neural networks (LSTM, if used)

---

## 📊 Visualizations Used

- Word clouds of common positive and negative words
- Distribution plots of review lengths
- Bar plots showing class balance
- Confusion matrix to evaluate model performance
- ROC-AUC curve for model comparison
![image](https://github.com/user-attachments/assets/5ccc73c5-7103-4e83-b371-77b82f28ad27)
![image](https://github.com/user-attachments/assets/26349780-c029-4a72-a6f9-6bb23cd566bd)

---

## 🙋 Author

**Rahul Yadav**  
Aspiring Data Scientist | NLP Enthusiast | Machine Learning Practitioner  
📫 [LinkedIn](https://www.linkedin.com/in/rahulyadav2707/)
