from datasets import load_dataset, load_metric
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments, Trainer

# Pre process the data (pair of sentences)
raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


# Instantiate the trainer. TrainingArguments class that will contain all the hyperparameters
# the Trainer will use for training and evaluation
training_args = TrainingArguments("test-trainer", evaluation_strategy="epoch")

# Now we need to define our model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

# Define a compute metrics
def compute_metrics(eval_preds):
    metric = load_metric("glue", "mrpc")
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Once we have our model, we can define a Trainer by passing it all the objects constructed up to now — 
# the model, the training_args, the training and validation datasets, our data_collator, and our tokenizer
trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# Finally its time to train the model, Watch out for the GPUs to crank up.
trainer.train()

# Look at some predictions
predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)