# Task 2: Train the Model on Kaggle & Track with W&B
# Task 3: Evaluate & Save Results
# Task 4: Save Model to Hugging Face Hub

import json
import wandb

from huggingface_hub import login

from transformers import (
    TrainingArguments,
    Trainer
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)

# 1. Initialize W&B
wandb.init(
    project="mlops-assignment2",
    name="distilbert-run-1",

    config={
        "model": model_name,
        "epochs": 3,
        "batch_size": 16,
        "learning_rate": 3e-5,
        "max_length": max_length,
        "dataset": "UCSD Goodreads",
        "platform": "Kaggle",
    }
)

# 2. Define evaluation metrics
def compute_metrics(pred):

    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

# 3. Training arguments
training_args = TrainingArguments(

    output_dir="./results",

    num_train_epochs=3,

    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,

    warmup_steps=100,

    weight_decay=0.01,

    logging_steps=50,

    evaluation_strategy="epoch",

    save_strategy="epoch",

    load_best_model_at_end=True,

    report_to="wandb",

    run_name="distilbert-run-1",
)

# 4. Trainer
trainer = Trainer(

    model=model,

    args=training_args,

    train_dataset=train_dataset,

    eval_dataset=test_dataset,

    compute_metrics=compute_metrics,
)

# 5. Train model
trainer.train()

# 6. Run evaluation
eval_results = trainer.evaluate()

print(eval_results)

# 7. Log final metrics to W&B
wandb.log({
    "final/loss": eval_results["eval_loss"],
    "final/accuracy": eval_results["eval_accuracy"],
})

# 8. Save full classification report
preds = trainer.predict(test_dataset).predictions.argmax(-1)

labels = [item["labels"].item() for item in test_dataset]

report = classification_report(
    labels,
    preds,
    target_names=list(id2label.values()),
    output_dict=True
)

with open("eval_report.json", "w") as f:
    json.dump(report, f, indent=2)

# 9. Upload to W&B as a versioned Artifact
artifact = wandb.Artifact(
    "eval-report",
    type="evaluation"
)

artifact.add_file("eval_report.json")

wandb.log_artifact(artifact)

# 10. Login to Hugging Face
login(token=HF_TOKEN)

# 11. Your Hugging Face repo name
repo_name = "Vishnu1010/distilbert-imdb-classifier"

# 12. Push model
model.push_to_hub(repo_name)

# 13. Push tokenizer
tokenizer.push_to_hub(repo_name)

# 14. Hugging Face model URL
model_url = f"https://huggingface.co/{repo_name}"

print("Model uploaded successfully!")
print("Model URL:", model_url)

# 15. Log URL to W&B
if wandb.run is not None:
    wandb.run.summary["huggingface_model"] = model_url

# 16. Finish W&B run
wandb.finish()
