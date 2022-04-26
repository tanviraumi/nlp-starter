from datasets import load_dataset
import html

# Load, shuffle, and take a pick at the datasets
data_files = {"train": "datasets/drugsComTrain_raw.tsv", "test": "datasets/drugsComTest_raw.tsv"}
# \t is the tab character in Python
drug_dataset = load_dataset("csv", data_files=data_files, delimiter="\t")

drug_sample = drug_dataset["train"].shuffle(seed=42).select(range(1000))
# Peek at the first few examples
print(drug_sample[:3])

# Change unnamed column to patient_id
drug_dataset = drug_dataset.rename_column(
    original_column_name="Unnamed: 0", new_column_name="patient_id"
)
print(drug_dataset)

# Normalize the condition column. But first we have to discard the empty entries
drug_dataset = drug_dataset.filter(lambda x: x["condition"] is not None)

def lowercase_condition(example):
    return {"condition": example["condition"].lower()}

drug_dataset = drug_dataset.map(lowercase_condition)

print(drug_dataset["train"]["condition"][:3])

# Add a review_length column an inspect again
def compute_review_length(example):
    return {"review_length": len(example["review"].split())}

drug_dataset = drug_dataset.map(compute_review_length)
print(drug_dataset["train"][0])

# Now filter out reviews to containe only reviews with less than 30 words
drug_dataset = drug_dataset.filter(lambda x: x["review_length"] > 30)
print(drug_dataset.num_rows)

drug_dataset = drug_dataset.map(
    lambda x: {"review": [html.unescape(o) for o in x["review"]]}, batched=True
)
