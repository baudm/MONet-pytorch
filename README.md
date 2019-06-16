# MONet in PyTorch

We provide a PyTorch implementation of [MONet](https://arxiv.org/abs/1901.11390).

This project is built on top of the CycleGAN/pix2pix code written by [Jun-Yan Zhu](https://github.com/junyanz) and [Taesung Park](https://github.com/taesung), and supported by [Tongzhou Wang](https://ssnl.github.io/).

**Note**: The current software works well with PyTorch 1.0+.

You may find useful information in [training/test tips](docs/tips.md) and [frequently asked questions](docs/qa.md). To implement custom models and datasets, check out our [templates](#custom-model-and-dataset). To help users better understand and adapt our codebase, we provide an [overview](docs/overview.md) of the code structure of this repository.

## Current Results
<img src="https://i.imgur.com/59x65ML.png" width="800"/>
<img src="https://i.imgur.com/HUuhdzC.png" width="800"/>

## Prerequisites
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/baudm/MONet-pytorch.git
cd MONet-pytorch
```

- Install [PyTorch](http://pytorch.org and) 1.0+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
  - For Conda users, we provide a installation script `./scripts/conda_deps.sh`. Alternatively, you can create a new Conda environment using `conda env create -f environment.yml`.
  - For Docker users, we provide the pre-built Docker image and Dockerfile. Please refer to our [Docker](docs/docker.md) page.

### MONet train/test
- Download a MONet dataset (e.g. CLEVR):
```bash
wget -cN https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0.zip
```
- To view training results and loss plots, run `python -m visdom.server` and click the URL http://localhost:8097.
- Train a model:
```bash
python train.py --dataroot ./datasets/CLEVR_v1.0 --name clevr_monet --model monet
```
To see more intermediate results, check out `./checkpoints/clevr_monet/web/index.html`.

### Apply a pre-trained model
- TODO
