import os
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import Dataset

# ----------------------------
# CONFIG
# ----------------------------
MODEL_NAME = "xlm-roberta-base"
DATA_PATH = "urdu_dataset.csv"
OUTPUT_DIR = "./fine_tuned_urdu_model"
MAX_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------------------
# LOAD DATA
# ----------------------------
df = pd.read_csv(DATA_PATH)

# Encode labels for multitask classification
label_encoders = {}
for col in ["sentiment", "topic", "intent", "binary"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

import numpy as np

encoded_labels = []
for row in df[["sentiment", "topic", "intent", "binary"]].values:
    one_hot = []
    idx_offset = 0
    for i, col in enumerate(["sentiment", "topic", "intent", "binary"]):
        num_classes = len(label_encoders[col].classes_)
        vec = [0] * num_classes
        vec[row[i]] = 1
        one_hot.extend(vec)
        idx_offset += num_classes
    encoded_labels.append(one_hot)

df["labels"] = encoded_labels

# Split dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["text"].tolist(),
    df["labels"].tolist(),
    test_size=0.1,
    random_state=42,
)

# ----------------------------
# TOKENIZATION
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=MAX_LENGTH)

train_dataset = Dataset.from_dict({"text": train_texts, "labels": train_labels})
val_dataset = Dataset.from_dict({"text": val_texts, "labels": val_labels})

train_dataset = train_dataset.map(tokenize, batched=True)
val_dataset = val_dataset.map(tokenize, batched=True)

# Remove unnecessary columns for Trainer
train_dataset = train_dataset.remove_columns(["text"])
val_dataset = val_dataset.remove_columns(["text"])

# ----------------------------
# MODEL
# ----------------------------
# For multitask, we create a model with 4 output heads combined as one multi-label classification
num_labels_per_task = [len(label_encoders[col].classes_) for col in ["sentiment", "topic", "intent", "binary"]]
total_labels = sum(num_labels_per_task)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=total_labels,
)

model.to(DEVICE)

# ----------------------------
# DATA COLLATOR
# ----------------------------
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ----------------------------
# TRAINING ARGUMENTS
# ----------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    learning_rate=2e-5,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=50,
    save_total_limit=2,
    push_to_hub=False,
)

# ----------------------------
# TRAINER
# ----------------------------
def compute_metrics(eval_pred):
    # Simple accuracy for each task
    logits, labels = eval_pred
    import numpy as np
    split_logits = np.split(logits, np.cumsum(num_labels_per_task)[:-1], axis=1)
    split_labels = np.split(labels, np.cumsum(num_labels_per_task)[:-1], axis=1)
    accs = []
    for logit, label in zip(split_logits, split_labels):
        preds = np.argmax(logit, axis=1)
        acc = (preds == label).mean()
        accs.append(acc)
    return {f"task_{i}_acc": a for i, a in enumerate(accs)}

from torch.nn import BCEWithLogitsLoss
import torch

def custom_compute_loss(model, inputs, return_outputs=False):
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    loss_fct = BCEWithLogitsLoss()

    # Ensure labels and logits are the same shape
    labels = labels.float()
    loss = loss_fct(logits, labels)

    return (loss, outputs) if return_outputs else loss

from torch.nn import BCEWithLogitsLoss
from transformers import Trainer

from torch.nn import BCEWithLogitsLoss
from transformers import Trainer

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        loss_fct = BCEWithLogitsLoss()
        labels = labels.float()

        loss = loss_fct(logits, labels)

        return (loss, outputs) if return_outputs else loss



trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,   # this is OK
)



# ----------------------------
# TRAIN
# ----------------------------
trainer.train()

# Save the fine-tuned model
trainer.save_model(OUTPUT_DIR)
print(f"Model saved in {OUTPUT_DIR}")
