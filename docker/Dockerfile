FROM nvidia/cuda:11.1-runtime-ubuntu20.04

# Prerequisites
## set timezone
RUN echo 'Etc/UTC' > /etc/timezone && \
    ln -s /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    apt-get update && apt-get install -q -y tzdata
    
## set noninteractive mode for installation
ENV DEBIAN_FRONTEND=noninteractive

## install packages
RUN apt-get update && apt-get install -q -y \
    dirmngr \
    gnupg2 \
    lsb-release \
    python3-venv \
    python3-pip \
    nano \
    vim \
    git

# disable pip version check    
ENV PIP_DISABLE_PIP_VERSION_CHECK=on

# poetry
RUN pip3 install --no-cache-dir poetry

# create working directory
WORKDIR /workspace

# clone project
RUN git clone https://github.com/CBMI-HTW/TaxonomicClassification-NGS-NN.git

# project related requirements
RUN cd TaxonomicClassification-NGS-NN && POETRY_VIRTUALENVS_CREATE=false poetry install --no-root --no-interaction --no-ansi

# torch cuda packages
RUN pip3 install --no-cache-dir torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# nvidia-container-runtime (for opengl support inside container)
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics,compat32,utility

# clean up
RUN rm -rf /var/lib/apt/lists/*

ENTRYPOINT cd /workspace/TaxonomicClassification-NGS-NN && python3 reproduce_results.py
