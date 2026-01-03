# transformer-vs-lstm-qa
# Transformer-based Question Answering on bAbI Dataset

This project implements a **Transformer-based Question Answering (QA) system** using the **bAbI dataset**.  
The goal is to understand how **self-attention and multi-head attention** improve reasoning over stories compared to traditional LSTM-based models.

---

## 📌 Project Objectives

- Prepare and preprocess the bAbI QA dataset  
- Implement a Transformer encoder with self-attention  
- Apply multi-head attention for story–question interaction  
- Train and evaluate the model  
- Compare Transformer performance with an LSTM baseline  

---

## 🧠 Dataset: bAbI Question Answering Tasks

The **bAbI dataset** is a synthetic QA dataset designed to test **logical reasoning and memory** in neural networks.

Each data sample consists of:
- **Story** – a sequence of facts
- **Question** – a query based on the story
- **Answer** – a single-word answer

---

## 🧩 Task Breakdown

### 🔹 Task 1: Dataset Preparation
- Download the bAbI dataset using **KaggleHub**
- Parse text files to extract:
  - Stories
  - Questions
  - Answers
- Build a vocabulary and map tokens to indices

---

### 🔹 Task 2: Self-Attention Encoder
- Tokenize and pad story and question sequences
- Convert tokens into embeddings
- Apply a **Transformer encoder** with:
  - Self-attention
  - Positional encoding
  - Feed-forward layers
- Encode contextual relationships within the story

---

### 🔹 Task 3: Multi-Head Attention for QA
- Encode **story** and **question** using embedding layers
- Use **question embeddings as Query**
- Use **story embeddings as Key and Value**
- Apply **multi-head attention** to focus on relevant story facts
- Aggregate attended features for answer prediction

---

### 🔹 Task 4: Training and Evaluation
- Train the Transformer-based QA model
- Evaluate performance on the test set
- Measure:
  - Training accuracy
  - Test accuracy
- Compare results with an **LSTM-based QA model**

---

## 🏗️ Model Architecture Overview

- **Embedding Layer**
- **Transformer Encoder (Self-Attention)**
- **Multi-Head Attention (Question → Story)**
- **Dense Output Layer**
- **Softmax for Answer Prediction**

---

## 📊 Results and Comparison

| Model Type | Test Accuracy | Reasoning Capability |
|-----------|--------------|---------------------|
| LSTM-based QA | Lower | Limited long-range dependency |
| Transformer-based QA | Higher | Strong global attention and reasoning |

**Observation:**  
The Transformer model outperforms LSTM due to its ability to:
- Capture long-term dependencies
- Focus on relevant facts using attention
- Avoid vanishing gradient issues

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- KaggleHub
- Google Colab / Kaggle Notebook

---

## 🚀 How to Run

1. Clone the repository
   ```bash
   git clone https://github.com/your-username/transformer-babi-qa.git
Open the notebook in Google Colab or Kaggle

Run all cells sequentially

View training logs and accuracy results