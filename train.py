from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, TrainingArguments

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
training_args = TrainingArguments("test-trainer")

# Now we need to define our model
model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)