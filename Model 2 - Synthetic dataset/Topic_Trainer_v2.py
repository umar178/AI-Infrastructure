# Import necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TextClassificationPipeline,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import Dataset
import torch
import evaluate  # For metrics

# Define project_dir
project_dir = os.getcwd()  # use current directory
model_save_dir = os.path.join(project_dir, "models", "fine_tuned_topic_classification_model")
os.makedirs(model_save_dir, exist_ok=True)

# Load dataset
df = pd.read_csv("urdu_dataset.csv").dropna()

# Encode labels (topic field)
unique_labels = df["topic"].unique()
label2id = {label: idx for idx, label in enumerate(unique_labels)}
id2label = {idx: label for label, idx in label2id.items()}
df["label"] = df["topic"].map(label2id)

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(test_df, test_size=0.5, random_state=42)

print(f"Train size: {len(train_df)}, Validation size: {len(val_df)}, Test size: {len(test_df)}")

# Load tokenizer and model
model_name = "Aimlab/xlm-roberta-base-finetuned-urdu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(unique_labels),
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True   # <-- important
)

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding=False
    )

# Convert to datasets
train_dataset = Dataset.from_pandas(train_df[['text', 'label']])
val_dataset = Dataset.from_pandas(val_df[['text', 'label']])
test_dataset = Dataset.from_pandas(test_df[['text', 'label']])

# Tokenize datasets
tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_val = val_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

# Data collator
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir='./logs',
    logging_steps=100,
    report_to="none"
)

# Metric function
accuracy_metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)

# Initialize trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Fine-tune the model
print("Starting fine-tuning...")
trainer.train()

# Save the fine-tuned model
trainer.save_model(model_save_dir)
tokenizer.save_pretrained(model_save_dir)
print("Fine-tuned model saved!")

# Evaluate on test set
print("Evaluating on test set...")
test_predictions = trainer.predict(tokenized_test)
test_metrics = test_predictions.metrics
print(f"Test accuracy: {test_metrics['test_accuracy']:.4f}")

# Classification report
predicted_labels = np.argmax(test_predictions.predictions, axis=1)
true_labels = test_df['label'].values
print("\nDetailed Classification Report:")
print(classification_report(true_labels, predicted_labels, target_names=list(label2id.keys())))

# Create prediction pipeline
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
fine_tuned_model = AutoModelForSequenceClassification.from_pretrained(model_save_dir).to(device)
fine_tuned_tokenizer = AutoTokenizer.from_pretrained(model_save_dir)

fine_tuned_pipe = TextClassificationPipeline(
    model=fine_tuned_model,
    tokenizer=fine_tuned_tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

# Test with sample text
sample_text = "یہ فلم بہت زبردست تھی"
result = fine_tuned_pipe(sample_text)
print(f"Sample prediction: {result}")
