import pickle
import os
import time
import shutil

import torch

import data
from vocab import Vocabulary  # NOQA
from model import VSE
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data

import logging
import tensorboard_logger as tb_logger

import argparse


def main():
    from vocab import Vocabulary
    import evaluation
    #evaluation.evalrank("runs/coco_vse++/model_best.pth.tar", data_path="data", split="test")
    evaluation.evalrank("runs/coco_vse++_vse/model_best.pth.tar", data_path="data", split="test")

if __name__ == '__main__':
    main()
