from __future__ import print_function

from miscc.config import cfg, cfg_from_file
from image_ae_trainer import IAETrainer as trainer
from image_ae_dataset import get_imgs, ImageDataset
import os
import sys
import time
import random
import pprint
import argparse
import numpy as np

import torch
import torchvision.transforms as transforms

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)


def parse_args():
    parser = argparse.ArgumentParser(description='Train a rewired image autoencoder network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/coco_attn2.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=-1)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--pretrained-cnn', dest='pretrained_cnn', type=str,
                        help='pretrained cnn encoder', default='')
    parser.add_argument('--pretrained-g', dest='pretrained_generator', type=str,
                        help='pretrained generator', default='')
    args = parser.parse_args()
    return args


def gen_example(dataset, algo):
    '''generate images from example images'''
    filepath = '%s/image_example_filenames.txt' % (cfg.DATA_DIR)
    data_dic = {}

    with open(filepath, "r") as f:
        filenames = f.read().decode('utf8').split('\n')
        for name in filenames:
            if len(name) == 0:
                continue
            filepath = '%s/image_ae_examples/%s.jpg' % (cfg.DATA_DIR, name)
            print('Load from:', name)
            image = get_imgs(filepath, dataset.imsize, bbox=None, transform=dataset.transform, normalize=dataset.norm)
            data_dic[name] = image[-1]
    algo.gen_example(data_dic)


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id != -1:
        cfg.GPU_ID = args.gpu_id
    else:
        cfg.CUDA = False

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir

    if args.pretrained_cnn:
        cfg.PRETRAINED_CNN = args.pretrained_cnn

    if args.pretrained_generator:
        cfg.PRETRAINED_G = args.pretrained_generator

    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = random.randint(1, 10000)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    split_dir, bshuffle = 'train', True
    if not cfg.TRAIN.FLAG:
        # bshuffle = False
        split_dir = 'test'

    # Get data loader
    imsize = cfg.TREE.BASE_SIZE * (2 ** (cfg.TREE.BRANCH_NUM - 1))
    image_transform = transforms.Compose([
        transforms.Scale(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    dataset = ImageDataset(cfg.DATA_DIR, split_dir,
                          base_size=cfg.TREE.BASE_SIZE,
                          transform=image_transform)
    assert dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=cfg.TRAIN.BATCH_SIZE,
        drop_last=True, shuffle=bshuffle, num_workers=int(cfg.WORKERS))

    # Define models and go to train/evaluate
    algo = trainer(dataloader)

    start_t = time.time()
    if cfg.TRAIN.FLAG:
        algo.train()
    else:
        '''generate images from pre-extracted embeddings'''
        if cfg.B_VALIDATION:
            algo.sampling(split_dir)  # generate images for the whole valid dataset
        else:
            gen_example(dataset, algo)  # generate images from customized images
    end_t = time.time()
    print('Total time for training:', end_t - start_t)
