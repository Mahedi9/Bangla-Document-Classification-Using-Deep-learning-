# Bangla Document Classification Using Deep learning
## Hybrid Ensemble of Bangla-BERT and Bi-LSTM

A research-grade Bangla news document classification system developed using a **Hybrid Ensemble of Bangla-BERT and Bi-LSTM models**.  
The system is deployed as an **interactive Streamlit web application** with **confidence-aware decision making** to handle overlapping news categories.

---

## Overview

This project performs automatic classification of Bangla news articles into predefined categories using deep learning models.  
It combines the contextual strength of **Bangla-BERT** with the sequential learning capability of **Bi-LSTM**, achieving robust and reliable predictions.

A confidence-threshold mechanism is used to detect ambiguous cases where multiple classes overlap.

---

## How to Run the Project

### 1. Clone the Repository

```bash
git clone https://github.com/Mahedi9/Bangla-Document-Classification-Using-Deep-learning-.git
cd Bangla-Document-Classification
```
This repository uses Git Large File Storage (LFS) for large model files.
Install Git LFS before cloning: https://git-lfs.github.com/

# Bangla Document Classification Using Hybrid Ensemble

A professional, research-grade Bangla news document classification system implemented using a **Hybrid Ensemble of Bangla-BERT and Bi-LSTM models**, deployed through an interactive **Streamlit web application** with confidence-aware decision making.

---

## How to Run the Project

### 1. Clone the Repository

Use the following command to clone the repository and enter the project directory:

git clone https://github.com/Mahedi9/Bangla-Document-Classification-Using-Deep-learning-.git  
cd Bangla-Document-Classification

This repository uses **Git Large File Storage (LFS)** for large model files.  
Please install Git LFS before cloning: https://git-lfs.github.com/

---

### 2. Install Dependencies

Install all required Python packages using:

pip install -r requirements.txt

---

### 3. Run the Streamlit Application

Launch the Streamlit application using:

streamlit run app.py

The application will open automatically in your default web browser.

---

## Application Features

- Bangla news article input through a web interface
- Separate predictions from:
  - **Bi-LSTM**
  - **Bangla-BERT**
- **Hybrid Ensemble prediction** using weighted soft voting  
  (0.6 × Bangla-BERT + 0.4 × Bi-LSTM)
- **Confidence-threshold based ambiguity detection** for overlapping classes
- Final decision explanation for low-confidence predictions
- Class-wise probability bar charts for model interpretability
- Clean, interactive, and research-oriented Streamlit user interface

---

## Supported News Categories

The classifier predicts one of the following **8 news categories**:

- Education  
- Entertainment  
- Economy  
- International  
- National  
- Politics  
- Science_Technology  
- Sports  

Predictions strictly follow the **Potrika Bangla News Dataset labeling policy**, which may differ from general human interpretation.

---

## Project Structure

Bangla-Document-Classification/  
├── app.py  
├── requirements.txt  
├── README.md    
├── .gitattributes  
│  
├── bilstm_model.keras  
├── bilstm_tokenizer.pkl  
├── bilstm_label_encoder.pkl  
│  
└── bangla_bert_model/  

---

## Model Files

This repository includes **pre-trained model files**, tracked using **Git LFS**:

- Bi-LSTM model (bilstm_model.keras)
- Tokenizer (bilstm_tokenizer.pkl)
- Label encoder (bilstm_label_encoder.pkl)
- Fine-tuned Bangla-BERT model directory (bangla_bert_model)

Ensure Git LFS is installed before cloning or pulling the repository.

---

## Research Paper

This project is based on the following **published research paper**:

**Title:**  
Enhancing Bangla Document Classification Using a Hybrid Ensemble of Bangla-BERT and Bi-LSTM Models

**Conference:**  
International Conference on Intelligent Data Analysis and Applications (IDAA 2025)

**Venue:**  
Daffodil International University, Dhaka, Bangladesh

**Dataset:**  
Potrika Bangla News Dataset  
329,110 Bangla news articles across 8 classes

**Hybrid Ensemble Test Accuracy:**  
97.16%

---

## Author

**Mahedi Hasan Emon**  
Published Author, IDAA 2025  
Bangla NLP | Deep Learning | Machine Learning

---

## Acknowledgements

- Potrika Bangla News Dataset  
- Hugging Face Transformers  
- TensorFlow  
- PyTorch  
- Daffodil International University  

---

## Notes

This repository represents a complete **research-to-deployment pipeline** for Bangla document classification and is suitable for academic projects, demonstrations, and further research extensions.
