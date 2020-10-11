import pickle
import os
import time
import shutil

import torch

import data
from vocab import Vocabulary  # NOQA
from model import XRN
from evaluation import i2t, t2i, AverageMeter, LogCollector, encode_data

import logging
import tensorboard_logger as tb_logger
import opts
import pdb
import numpy as np

def main(opt):
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)

    # Load Vocabulary Wrapper
    vocab = pickle.load(open(os.path.join(opt.vocab_path, 'vg_c_precomp_vocab.pkl'), 'rb'))
    opt.vocab_size = len(vocab)
    print(opt.vocab_size)

    # Load data loaders
    train_loader, val_loader = data.get_loaders(opt.data_name, vocab, opt.crop_size, opt.batch_size, opt.workers, opt)

    # Construct the model
    model = XRN(opt)

    # optionally resume from a checkpoint
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})".format(opt.resume, start_epoch, best_rsum))
            validate(opt, val_loader, model)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))

    # Train the Model
    best_rsum = 0
    for epoch in range(opt.num_epochs):
        adjust_learning_rate(opt, model.optimizer, epoch)

        # train for one epoch
        train(opt, train_loader, model, epoch, val_loader)

        # evaluate on validation set
        rsum, d_i2t = validate(opt, val_loader, model)

        # remember best R@ sum and save checkpoint
        is_best = rsum > best_rsum
        if is_best:
            np.save('d_i2t.npy',d_i2t)

        best_rsum = max(rsum, best_rsum)
        save_checkpoint({
            'epoch': epoch + 1,
            'model': model.state_dict(),
            'best_rsum': best_rsum,
            'opt': opt,
            'Eiters': model.Eiters,
        }, is_best, prefix=opt.logger_name + '_' + opt.model_name + '/')


def train(opt, train_loader, model, epoch, val_loader):
    # average meters to record the training statistics
    batch_time = AverageMeter()
    data_time = AverageMeter()
    train_logger = LogCollector()

    # switch to train mode
    model.train_start()

    end = time.time()
    for i, train_data in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        # make sure train logger is used
        model.logger = train_logger

        # Update the model
        model.train_emb(*train_data)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log info
        if model.Eiters % opt.log_step == 0:
            logging.info('Epoch:[{0}][{1}/{2}]{e_log}'.format(epoch, i, len(train_loader), e_log=str(model.logger)))

        # Record logs in tensorboard
        tb_logger.log_value('epoch', epoch, step=model.Eiters)
        tb_logger.log_value('step', i, step=model.Eiters)
        tb_logger.log_value('batch_time', batch_time.val, step=model.Eiters)
        tb_logger.log_value('data_time', data_time.val, step=model.Eiters)
        model.logger.tb_log(tb_logger, step=model.Eiters)

        # validate at every val_step
        if model.Eiters % opt.val_step == 0:
            validate(opt, val_loader, model)


def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    img_embs, cap_embs = encode_data(model, val_loader, opt.log_step, logging.info)

    # caption retrieval
    (r1, r5, r10, medr, meanr), d_i2t = i2t(img_embs, cap_embs, measure=opt.measure)
    
    logging.info("Image to text: %.1f, %.1f, %.1f, %.1f, %.1f" % (r1, r5, r10, medr, meanr))
    # image retrieval
    (r1i, r5i, r10i, medri, meanr), d_t2i = t2i(img_embs, cap_embs, measure=opt.measure)
    logging.info("Text to image: %.1f, %.1f, %.1f, %.1f, %.1f" %  (r1i, r5i, r10i, medri, meanr))
    # sum of recalls to be used for early stopping
    currscore = r1 + r5 + r10 + r1i + r5i + r10i

    # record metrics in tensorboard
    tb_logger.log_value('recall@1_text', r1, step=model.Eiters)
    tb_logger.log_value('recall@5_text', r5, step=model.Eiters)
    tb_logger.log_value('recall@10_text', r10, step=model.Eiters)
    tb_logger.log_value('med-r_text', medr, step=model.Eiters)
    tb_logger.log_value('mean-r_text', meanr, step=model.Eiters)
    tb_logger.log_value('recall@1_im', r1i, step=model.Eiters)
    tb_logger.log_value('recall@5_im', r5i, step=model.Eiters)
    tb_logger.log_value('recall@10_im', r10i, step=model.Eiters)
    tb_logger.log_value('med-r_im', medri, step=model.Eiters)
    tb_logger.log_value('mean-r_im', meanr, step=model.Eiters)
    tb_logger.log_value('recall_sum', currscore, step=model.Eiters)

    return currscore, d_i2t


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', prefix=''):
    if not os.path.exists(prefix):
        os.makedirs(prefix)
    torch.save(state, prefix + filename)
    if is_best:
        shutil.copyfile(prefix + filename, prefix + 'model_best.pth.tar')


def adjust_learning_rate(opt, optimizer, epoch):
    """Sets the learning rate to the initial LR
       decayed by 10 every 30 epochs"""
    lr = opt.learning_rate * (0.1 ** (epoch // opt.lr_update))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


if __name__ == '__main__':
    opt = opts.parse_opt()
    main(opt)
