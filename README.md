# MLOps Assignment: DistilBERT Text Classification Pipeline

## Overview
This project demonstrates a complete MLOps workflow using Hugging Face Transformers, Kaggle GPU training, Weights & Biases (W&B) experiment tracking, and Hugging Face Hub deployment. The model used in this assignment is `distilbert-base-cased`, which is a lightweight and faster version of BERT suitable for text classification tasks.

The workflow includes:
- Loading a pre-trained DistilBERT model from Hugging Face
- Fine-tuning the model on a classification dataset using Kaggle GPU
- Tracking experiments and metrics with W&B
- Evaluating model performance using Accuracy and F1 Score
- Saving evaluation reports as W&B artifacts
- Uploading the trained model and tokenizer to Hugging Face Hub

---

# Task 2: Load a Pre-trained Model from Hugging Face

## Description
This script loads the `distilbert-base-cased` tokenizer and model from Hugging Face Transformers. The model is configured for binary classification with 2 output labels.

## Key Features
- Uses `DistilBertTokenizerFast`
- Loads `DistilBertForSequenceClassification`
- Runs on GPU using CUDA
- Supports maximum sequence length of 512

## Model Details
- Model Name: `distilbert-base-cased`
- Output Labels: `2`
- Device: `CUDA GPU`

---

# Task 2: Train the Model on Kaggle & Track with W&B

## Description
This script fine-tunes the DistilBERT model using Hugging Face Trainer API inside a Kaggle Notebook with GPU acceleration. Weights & Biases (W&B) is integrated for experiment tracking.

## Features
- W&B experiment tracking
- Accuracy and weighted F1-score computation
- GPU training on Kaggle
- Automatic evaluation after training
- Hyperparameter logging

## Training Configuration

| Parameter | Value |
|---|---|
| Epochs | 3 |
| Train Batch Size | 16 |
| Eval Batch Size | 32 |
| Learning Rate | 3e-5 |
| Weight Decay | 0.01 |
| Logging Steps | 50 |

## W&B Tracking
The following metrics are logged:
- Training Loss
- Validation Loss
- Accuracy
- F1 Score
- Hyperparameters
- Hugging Face Model URL

---

# Task 3: Evaluate & Save Results

## Description
After training, the model is evaluated on the test dataset.

## Evaluation Metrics
- Accuracy
- F1 Score
- Evaluation Loss

## Additional Features
- Generates a detailed classification report
- Saves report as `eval_report.json`
- Uploads evaluation report to W&B as an artifact

---

# Task 4: Save Model to Hugging Face Hub

## Description
The trained DistilBERT model and tokenizer are uploaded to Hugging Face Hub for public access.

## Hugging Face Repository
https://huggingface.co/Vishnu1010/distilbert-imdb-classifier

---

# GitHub Repository
https://github.com/justCodeItVishnu/mlOps

---

# W&B Dashboard
https://wandb.ai/g25ait2128-prom-iit-rajasthan/mlops-assignment2?nw=nwuserg25ait2128

---

# Requirements

```bash
pip install transformers datasets torch wandb scikit-learn huggingface_hub
```

---

# Running the Project

## Step 1: Load Model
Run the model loading script.

## Step 2: Train the Model
Run the Kaggle training script with GPU enabled.

## Step 3: Evaluate
The script automatically evaluates the model after training.

## Step 4: Upload Model
The trained model and tokenizer are pushed to Hugging Face Hub.

---

# Results

| Metric | Score |
|---|---|
| Accuracy | Logged in W&B |
| F1 Score | Logged in W&B |
| Eval Loss | Logged in W&B |

---

# Conclusion
This assignment demonstrates an end-to-end MLOps pipeline using Hugging Face Transformers, Kaggle GPU infrastructure, W&B experiment tracking, and Hugging Face Hub deployment. The workflow ensures reproducibility, model monitoring, evaluation tracking, and public model sharing in a production-style setup.
