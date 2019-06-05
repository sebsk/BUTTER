# ImageCaptioning_pretrain

This folder is modified from Ruotian Luo's `ImageCaptioning.Pytorch` repository at [link] https://github.com/ruotianluo/ImageCaptioning.pytorch

## Requirements
Python 2.7 (because there is no [coco-caption](https://github.com/tylin/coco-caption) version for python 3)
PyTorch 0.4.1 (along with torchvision)

## Modification
LSTM unit cell weights and word embeddings learned from `text_autoencoder_pretrain` are imported as initialized weights for supervised fine-tuning. **ONLY SUPPORT ATT2IN MODEL**
