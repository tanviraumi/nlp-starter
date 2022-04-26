from datasets import load_dataset, Dataset

# Load the issues+comment dataset and filter out PRs and entires without comment
issues_dataset = load_dataset("json", data_files="/home/tanvir/work/nlp-starter/datasets/github/issues-datasets-with-comments.jsonl", split="train")

issues_dataset = issues_dataset.filter(
    lambda x: (x["is_pull_request"] == False and len(x["comments"]) > 0)
)
print(issues_dataset)

# From a search perspective, the most informative columns are title, body, and comments, while html_url provides us with a link back to the source issue.
columns = issues_dataset.column_names
columns_to_keep = ["title", "body", "html_url", "comments"]
columns_to_remove = set(columns_to_keep).symmetric_difference(columns)
issues_dataset = issues_dataset.remove_columns(columns_to_remove)
print(issues_dataset)

# Because our comments column is currently a list of comments for each issue, we need to “explode” the column so that each row consists of an (html_url, title, body, comment) tuple.
# In Pandas we can do this with the DataFrame.explode() function, which creates a new row for each element in a list-like column, while replicating all the other column values.
issues_dataset.set_format("pandas")
df = issues_dataset[:]
comments_df = df.explode("comments", ignore_index=True)

# Now switch back to dataframe
comments_dataset = Dataset.from_pandas(comments_df)

# let’s create a new comments_length column that contains the number of words per comment. We can use this new column to filter out short comments
# Along with that, we also filter out None types
comments_dataset = comments_dataset.map(
    lambda x: {"comment_length": len(x["comments"].split())}
)
comments_dataset = comments_dataset.filter(lambda x: x["title"] is not None and x["body"] is not None and x["comment_length"] > 15)

# let’s concatenate the issue title, description, and comments together in a new text column.
def concatenate_text(examples):
    return {
        "text": examples["title"]
        + " \n "
        + examples["body"]
        + " \n "
        + examples["comments"]
    }
comments_dataset = comments_dataset.map(concatenate_text)
print(comments_dataset)

# Now dump the data
comments_dataset.to_json("/home/tanvir/work/nlp-starter/datasets/github/dataset-for-search.jsonl")
 