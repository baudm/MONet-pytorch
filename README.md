# MONet in PyTorch

We provide a PyTorch implementation of [MONet](https://arxiv.org/abs/1901.11390).

This project is built on top of the [CycleGAN/pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) code written by [Jun-Yan Zhu](https://github.com/junyanz) and [Taesung Park](https://github.com/taesung), and supported by [Tongzhou Wang](https://ssnl.github.io/).

**Note**: The implementation is developed and tested on Python 3.7 and PyTorch 1.1.

## Test Results
### CLEVR 64x64 @ 160 epochs
<img src="https://i.imgur.com/wjIyVhe.png" width="748"/>
<img src="https://i.imgur.com/qFYkglK.png" width="748"/>

## Prerequisites
- Linux or macOS (not tested)
- Python 3.7
- CPU or NVIDIA GPU + CUDA 10 + CuDNN

## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/baudm/MONet-pytorch.git
cd MONet-pytorch
```

- Install [PyTorch](http://pytorch.org and) 1.1+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
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

To generate a montage of the model outputs like the ones shown above:
```bash
./scripts/test_monet.sh
./scripts/generate_monet_montage.sh
```

### Apply a pre-trained model
- Download pretrained weights for CLEVR 64x64:
```bash
./scripts/download_monet_model.sh clevr
```
