###
### Base container image with common tools and user accounts
###
FROM nvidia/cuda:11.2.2-base-ubuntu20.04 as base

# Temporary workaround cause NVIDIA is rotating their keys
# https://forums.developer.nvidia.com/t/notice-cuda-linux-repository-key-rotation/212772
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

# Python installation
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        python3 \
        python3-pip \
        python3-venv \
        python3-dev \
        procps

# Tini
ENV TINI_VERSION v0.19.0
ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /tini
RUN chmod +x /tini

# Non-root user to run under
RUN groupadd nlp-starter && \
    useradd --gid nlp-starter --create-home nlp-starter

# switch to non-root user
USER nlp-starter
RUN mkdir -p /home/nlp-starter/app
WORKDIR /home/nlp-starter/app

COPY --chown=nlp-starter:nlp-starter cuda.py .

RUN python3 -m venv .env && \
    . .env/bin/activate && \
    pip3 install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu113

