## Create and activate the virtual environment

```
python3 -m venv .env

source .env/bin/activate
```

## Install required packaages

Install Huggingface transformes
```
pip3 install "transformers[sentencepiece]"
```

Install torch based on your [architecture](https://pytorch.org/get-started/locally/)
```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

Now install Huggingface datasets
```
pip3 install datasets
```