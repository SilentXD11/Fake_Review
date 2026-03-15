# Fake Product Review Detection using BERT

## Overview

This project implements an AI-based system to detect **fake or deceptive product reviews** using Natural Language Processing and deep learning. Online platforms often contain fabricated reviews created to manipulate customer perception and influence purchasing decisions. This project aims to automatically identify such deceptive reviews using a transformer-based language model.

The system is built using **Python**, **PyTorch**, and **Hugging Face Transformers**, and utilizes the pretrained **BERT** model for text classification.

After training, the system allows users to input a review and instantly determine whether it is **Genuine or Fake**.

---

# Features

* Detects fake and genuine product reviews
* Uses a transformer-based deep learning model
* Fine-tunes a pretrained BERT model
* Performs text preprocessing and tokenization
* Interactive review classification
* Achieves high accuracy on labeled review data

---

# Dataset

This project uses the **Deceptive Opinion Spam Corpus dataset**, which contains **1600 hotel reviews**.

Dataset composition:

* 800 truthful reviews
* 800 deceptive reviews

Each review is labeled to indicate whether it is **genuine or fake**, making it suitable for binary classification tasks.

---

# Technologies Used

* Python
* PyTorch
* Hugging Face Transformers
* BERT
* pandas
* scikit-learn
* Jupyter Notebook

---

# Project Structure

```
fake-review-detection
│
├── data
│   └── deceptive-opinion.csv
│
├── notebooks
│   └── fake_review_detection.ipynb
│
├── model
│   └── trained_model
│
├── predict.py
├── train.py
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository:

```
git clone https://github.com/yourusername/fake-review-detection.git
```

Navigate to the project directory:

```
cd fake-review-detection
```

Install dependencies:

```
pip install -r requirements.txt
```

---

# Running the Project

## Train the Model

Run the training script:

```
python train.py
```

The BERT model will be fine-tuned using the labeled dataset.

---

## Predict Fake Reviews

Run:

```
python predict.py
```

Then enter a review:

```
Enter your review:
This hotel was amazing and the staff were very friendly
```

Output:

```
Prediction: Genuine Review
```

---

# Example Prediction

Input:

```
This hotel was the best I have ever stayed in and I highly recommend it
```

Output:

```
Fake Review
```

---

# Machine Learning Pipeline

The system follows the following pipeline:

1. **Data Collection**

   Reviews are collected from the Deceptive Opinion Spam Corpus dataset.

2. **Data Preprocessing**

   * Lowercasing text
   * Removing special characters
   * Cleaning unwanted tokens

3. **Tokenization**

   Reviews are tokenized using the **BERT tokenizer**.

4. **Model Training**

   A pretrained **BERT sequence classification model** is fine-tuned using labeled review data.

5. **Evaluation**

   Model performance is evaluated using a validation dataset.

6. **Inference**

   Users can input new reviews and receive predictions.

---

# Model Performance

Typical performance using BERT:

| Metric    | Score |
| --------- | ----- |
| Accuracy  | ~90%  |
| Precision | ~89%  |
| Recall    | ~90%  |

Performance may vary depending on preprocessing and training configuration.

---

# Example Workflow

```
User enters review
        ↓
Text preprocessing
        ↓
Tokenization with BERT
        ↓
Deep learning model inference
        ↓
Prediction: Fake or Genuine
```

---

# Applications

Fake review detection systems can be used in:

* E-commerce platforms
* Hotel booking websites
* Product review websites
* Social media moderation
* Online reputation monitoring

---

# Future Improvements

Possible extensions include:

* Training with larger review datasets
* Implementing **multi-platform review detection**
* Building a **web interface using Flask**
* Adding explainable AI to highlight suspicious phrases
* Deploying the model as an API

---

# Author

Master's student specializing in **Artificial Intelligence and Natural Language Processing**, with interests in deep learning, computer vision, and intelligent systems.

---

# License

This project is released under the **MIT License**.


