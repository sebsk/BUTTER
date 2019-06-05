# AttnGAN

This folder is modified from [link]https://github.com/taoxugit/AttnGAN.

## Dependencies
python 2.7

Pytorch

In addition, please add the project folder to PYTHONPATH and `pip install` the following packages:
- `python-dateutil`
- `easydict`
- `pandas`
- `torchfile`
- `nltk`
- `scikit-image`

## Modification
`image_ae_*.py` are for image autoencoder pre-training. `image_ae_pretrain.py` is for $IAE_{resnet}$ and the rest are for $IAE_{attngan}$. The two versions of autoencoder pre-training are introduced in our report.

More details about the modification are included in `changes.txt`.
