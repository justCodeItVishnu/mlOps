# Task 2: Load a Pre-trained Model from Hugging Face

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification
)

import torch

# Pre-trained model name
model_name = "distilbert-base-cased"

# Maximum token length
max_length = 512

# Number of output labels/classes
num_labels = 2

# Device configuration
device = torch.device("cuda")

# Load tokenizer
tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

# Load model
model = DistilBertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels
).to(device)

print("Tokenizer Loaded Successfully")
print("Model Loaded Successfully")
print("Using Device:", device)
