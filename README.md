# 🤖 AI-Based Detection of Fake and Deepfake Content on Social Media

## 📘 Project Overview
This project explores the **role of Artificial Intelligence (AI)** in the creation and detection of **fake content and deepfakes** across local social media platforms. It was conducted as part of an internship in the media and digital verification domain.

The project involves:
- Literature review of existing AI approaches.
- Collection and annotation of real-world fake content
- Building and testing an AI model using Python for classification and detection.
- Publishing the findings and model on Kaggle

---

## 🎯 Research Question

> How effective is Artificial Intelligence in detecting and combating the spread of fake and deepfake content on local social media platforms?

---

## 🔍 Research Activities

### 1. 🧠 Learning & Literature Review
- Summarized key scientific papers on fake/deepfake detection.
- Reviewed technologies used in content manipulation (GANs, deepfakes).
- Analyzed prior datasets and evaluation metrics.

### 2. 📊 Data Collection & Analysis
- Collected samples of fake content from local student social platforms.
- Verified examples using manual and automated methods.
- Categorized content into:
  - Text-based misinformation
  - Image/video deepfakes
  - Memes with false claims

### 3. 🤖 AI Model Development (Python)
- Preprocessed text and images.
- Used **BERT embeddings** for Arabic content and **EfficientNet** for video frames.
- Trained a classification model using **XGBoost/LightGBM / CatBoost**.
- Evaluated using accuracy, F1-score, and confusion matrix.

> 📌 Model published on Kaggle – see Model & Notebook here
Kaggle : https://www.kaggle.com/code/abdelrazzaqasfour/abdelrazzaq-m-asfour-7-9

---

## 🧠 Proposed Solution

A **hybrid AI detection framework**:
- NLP-based analysis for fake news detection (e.g., AraBERT)
- CNN-based analysis for image and video content
- Real-time flagging of suspicious content based on confidence score
- Integration into a browser extension or moderation tool

---

## 📽️ Deliverables

- ✔️ AI detection notebook on Kaggle
- ✔️  verified fake content examples (collected via Google Form)
- ✔️ Recorded video presentation with findings

---

## 📘 License

This project is for academic and research purposes only. 

---

Model selection and justifications

In terms of classifications, CatBoost achieved the highest accuracy in identifying Social Media. We also observe that it obtained the highest recall score. Although CatBoost had a longer training time, we do not want to sacrifice accuracy. Looking at the confusion matrix, CatBoost achieved the highest accuracy, which was 78.16%. It also recorded the highest precision and recall scores, which is evidence of its balanced performance in minimizing errors.

choose the CatBoost model

Accuracy |95.18% | Overall correctness. Higher is better.

Precision (weighted) 0.9059 Weighted average precision. Higher is better.

Recall (weighted) 0.9518 Weighted average recall. Higher is better.
