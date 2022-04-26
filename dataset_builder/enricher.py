import os
import requests
from datasets import load_dataset

issues_dataset = load_dataset("json", data_files="/home/tanvir/work/nlp-starter/datasets/github/datasets-issues.jsonl", split="train")
print(issues_dataset)

# each pull request is associated with various URLs, while ordinary issues have a None entry. 
# We can use this distinction to create a new is_pull_request column that checks whether the pull_request field is None or not:
issues_dataset = issues_dataset.map(
    lambda x: {"is_pull_request": False if x["pull_request"] is None else True}
)

GITHUB_TOKEN = os.environ["GITHUB_PAT_TO_FETCH"]
headers = {"Authorization": f"token {GITHUB_TOKEN}"}

# Now we need to enrich the data. Fetch comments for each issues and add it to the dataset.
def get_comments(issue_number):
    url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
    response = requests.get(url, headers=headers)
    return [r["body"] for r in response.json()]

issues_with_comments_dataset = issues_dataset.map(
    lambda x: {"comments": get_comments(x["number"])}
)

# Now dump the data
issues_with_comments_dataset.to_json("/home/tanvir/work/nlp-starter/datasets/github/issues-datasets-with-comments.jsonl")