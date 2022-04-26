from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd

# Load the dataset for semantic search
input_dataset = load_dataset("json", data_files="/home/tanvir/work/nlp-starter/datasets/github/dataset-for-search.jsonl", split="train")

# our use case is an example of asymmetric semantic search because we have a short query whose answer we’d like to find in a longer document, like a an issue comment.
# multi-qa-mpnet-base-dot-v1 checkpoint has the best performance for semantic search, so we’ll use that for our application. We’ll also load the tokenizer using the same checkpoint
model_ckpt = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)
model = AutoModel.from_pretrained(model_ckpt)

# To speed up the embedding process, it helps to place the model and inputs on a GPU device, so let’s do that now
device = torch.device("cuda")
model.to(device)

# we’d like to represent each entry in our GitHub issues corpus as a single vector, so we need to “pool” or average our token embeddings in some way.
# One popular approach is to perform CLS pooling on our model’s outputs, where we simply collect the last hidden state for the special [CLS] token
def cls_pooling(model_output):
    return model_output.last_hidden_state[:, 0]

# Next, we’ll create a helper function that will tokenize a list of documents, place the tensors on the GPU, feed them to the model, and finally apply CLS pooling to the outputs
def get_embeddings(text_list):
    encoded_input = tokenizer(
        text_list, padding=True, truncation=True, return_tensors="pt"
    )
    encoded_input = {k: v.to(device) for k, v in encoded_input.items()}
    model_output = model(**encoded_input)
    return cls_pooling(model_output)

# Now we are ready to use the Dataset.map() to apply our get_embeddings() function to each row in our corpus, so let’s create a new embeddings column as follows.
# Notice that we’ve converted the embeddings to NumPy arrays — that’s because Datasets requires this format when we try to index them with FAISS
embeddings_dataset = input_dataset.map(
    lambda x: {"embeddings": get_embeddings(x["text"]).detach().cpu().numpy()[0]}
)

# The basic idea behind FAISS is to create a special data structure called an index that allows one to find which embeddings are similar to an input embedding.
# Creating a FAISS index in Datasets is simple
embeddings_dataset.add_faiss_index(column="embeddings")

# We are now ready to perform queries. But first we need to embed the incoming query
question = "How can I load a dataset offline?"
question_embedding = get_embeddings([question]).cpu().detach().numpy()

# we now have a 768-dimensional vector representing the query, which we can compare against the whole corpus to find the most similar embeddings
scores, samples = embeddings_dataset.get_nearest_examples(
    "embeddings", question_embedding, k=5
)

# The Dataset.get_nearest_examples() function returns a tuple of scores that rank the overlap between the query and the document, 
# and a corresponding set of samples (here, the 5 best matches). Let’s collect these in a pandas.DataFrame so we can easily sort them
samples_df = pd.DataFrame.from_dict(samples)
samples_df["scores"] = scores
samples_df.sort_values("scores", ascending=False, inplace=True)

# Finally we can iterate over the first few rows to see how well our query matched the available comments
for _, row in samples_df.iterrows():
    print(f"COMMENT: {row.comments}")
    print(f"SCORE: {row.scores}")
    print(f"TITLE: {row.title}")
    print(f"URL: {row.html_url}")
    print("=" * 50)
    print()