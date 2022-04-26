```
pip3 install ipywidgets
pip3 install faiss-gpu
```

Generate a [Github PAT](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/creating-a-personal-access-token). Copy it and expose into shell:

```
export GITHUB_PAT_TO_FETCH="<your_PAT>"
```

1. **fetcher.py** -> Fetches github issues and stores them in **datasets/github** folder
2. **enricher.py** -> Enriches each issue with comments and stores then in **datasets/github** folder
3. **prepare.py** -> Prepares the dataset by adding a concatenated text column and stores then in **datasets/github** folder
4. **search.py** -> Runs semantic search over the dataset generated at step 3.

