# Fraud Detection System

## 📌 Project Overview
This project is an end-to-end machine learning pipeline for **credit card fraud detection**, built to handle highly **imbalanced datasets**.  
The goal is to identify fraudulent transactions with high accuracy while minimizing false positives.  

The project demonstrates **data preprocessing, machine learning, deep learning with PyTorch, and handling imbalanced data**, all while following best practices in data science project structure.

---

## 🚀 Features & Highlights
- **Data Preprocessing**
  - Cleaned raw credit card transaction data (removed unnecessary columns, normalized features).
  - Applied **train-test split** with standardized scaling for balanced evaluation.
  
- **Handling Class Imbalance**
  - Explored strategies like **oversampling** (Random Over Sampling & SMOTE).
  - Planned use of **weighted binary cross-entropy loss** to ensure fair learning across fraud/non-fraud classes.

- **Neural Network Model**
  - Implemented a custom **PyTorch neural network** with multiple hidden layers.
  - Trained using **Adam optimizer** and **Binary Cross-Entropy with Logits Loss**.
  - Monitored loss curves to assess learning progress.

- **Experimentation**
  - Adjusted model architectures (number of nodes & layers).
  - Evaluated impact of oversampling vs. weighted losses.
  - Tracked fraud vs. non-fraud class ratios throughout training.
  - Incorporated **SMOTE** and compare against Random Oversampling. 

- **Visualization**
  - Plotted loss curves during training to diagnose underfitting/overfitting.
  - Prepared for future evaluation with metrics like **precision, recall, F1-score, ROC-AUC**.

---

## 📊 Next Steps 
- Fine-tune the **model architecture and hyperparameters**.  
- Evaluate performance using **precision/recall tradeoffs** and **confusion matrices**.  
- Deploy the model via a **Flask API**, connected to a **MySQL database** for real-time predictions.  
- Build an **interactive dashboard** for fraud monitoring (e.g., with **Streamlit or Power BI**).  

---

## 🛠️ Tech Stack
- **Python**: PyTorch, scikit-learn, pandas, numpy, matplotlib, imbalanced-learn  
- **SQL**: MySQL (planned for deployment)  
- **FastAPI** (planned for deployment)  
- **Version Control**: Git/GitHub  



