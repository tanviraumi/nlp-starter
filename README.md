## Create and activate the virtual environment

```
python3 -m venv .env

source .env/bin/activate
```

## Install required packaages

Install torch based on your [architecture](https://pytorch.org/get-started/locally/)

```
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
```

Usually the command should work but in my case, Python version 3.10 was incompatible with Cuda so I had to install the nightly build instead.

```
pip3 install --pre torch torchvision torchaudio -f https://download.pytorch.org/whl/nightly/cu113/torch_nightly.html
```

Finally Check whether architecture and Cuda versions are installed properly

```
python3 cuda.py
```

Now install Huggingface transformes and datasets
```
pip3 install "transformers[sentencepiece]"
pip3 install datasets
```

Some projects (train.py) also requires scipy and sklearn

```
pip3 install scipy sklearn
```
