import pickle
import os
import time
import shutil

import torch

from model import UDAG

import logging
import tensorboard_logger as tb_logger
import opts
import pdb
import data
from unit import *
import numpy as np

def main(opt):
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    tb_logger.configure(opt.logger_name, flush_secs=5)



    # Construct the model
    model = UDAG(opt)

    # optionally resume from a checkpoint
    if opt.evaluation:
        val_loader = data.get_test_loader(opt.data_name, opt.batch_size, opt.workers, opt)
        
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            start_epoch = checkpoint['epoch']
            best_rsum = checkpoint['best_rsum']
            model.load_state_dict(checkpoint['model'])
            # Eiters is used to show logs as the continuation of another training
            model.Eiters = checkpoint['Eiters']
            print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})".format(opt.resume, start_epoch, best_rsum))
            _, sims = validate(opt, val_loader, model)
            np.save(opt.data_name+'_sims',sims)
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
    # if opt.resume:
    #     if os.path.isfile(opt.resume):
    #         print("=> loading checkpoint '{}'".format(opt.resume))
    #         checkpoint = torch.load(opt.resume)
    #         start_epoch = checkpoint['epoch']
    #         best_rsum = checkpoint['best_rsum']
    #         model.load_state_dict(checkpoint['model'])
    #         # Eiters is used to show logs as the continuation of another training
    #         model.Eiters = checkpoint['Eiters']
    #         print("=> loaded checkpoint '{}' (epoch {}, best_rsum {})".format(opt.resume, start_epoch, best_rsum))
    #         validate(opt, val_loader, model)
    #     else:
    #         print("=> no checkpoint found at '{}'".format(opt.resume))
    else:
        # Train the Model
        # Load data loaders
        train_loader, val_loader = data.get_loaders(opt.data_name, opt.batch_size, opt.workers, opt)
        best_rsum = 0
        for epoch in range(opt.num_epochs):
            adjust_learning_rate(opt, model.optimizer, epoch)

            # rsum = validate(opt, val_loader, model)

            # train for one epoch
            train(opt, train_loader, model, epoch, val_loader)

            # evaluate on validation set
            rsum = validate(opt, val_loader, model)

            # remember best R@ sum and save checkpoint
            is_best = rsum > best_rsum
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
    # for i, (subFea,objFea,preFea,subject_ind,object_ind,predicate_ind, \
    #     subject_boxe,object_boxe,im_info, subTxt,objTxt,preTxt) in enumerate(train_loader):
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
        # if model.Eiters % opt.val_step == 0:
        #     validate(opt, val_loader, model)


def validate(opt, val_loader, model):
    # compute the encoding for all the validation images and captions
    
    img_embs, cap_embs, cap_lens = encode_data(model, val_loader, opt)
    # img_cls_sub, img_cls_obj, img_cls_phr, cap_cls_sub, cap_cls_obj, cap_cls_phr, cap_cls_objects = all_clss[:]
    # (r1, r5, r10, r1i, r5i, r10i) = re_test_lab(img_cls_sub, cap_cls_sub, img_cls_obj, cap_cls_obj, \
    #                                 img_cls_phr, cap_cls_phr, cap_cls_objects, measure='cosine')

    img_embs = numpy.array([img_embs[i] for i in range(0, len(img_embs), 5)])#img cut 5
    # pdb.set_trace()
    start = time.time()
    if opt.cross_attn == 't2i':
        sims = shard_xattn_t2i(img_embs, cap_embs, cap_lens, opt, shard_size=128)
    elif opt.cross_attn == 'i2t':
        sims = shard_xattn_i2t(img_embs, cap_embs, cap_lens, opt, shard_size=128)
    else:
        raise NotImplementedError
    end = time.time()
    print("calculate similarity time:", end-start)

    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(img_embs, cap_embs, cap_lens, sims)

    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(
        img_embs, cap_embs, cap_lens, sims)


    currscore = r1 + r5 + r10 + r1i + r5i + r10i
    print("Image to text: %.1f, %.1f, %.1f" % (r1, r5, r10))
    print("Text to image: %.1f, %.1f, %.1f" % (r1i, r5i, r10i))
    # record metrics in tensorboard
    tb_logger.log_value('recall@1_text', r1, step=model.Eiters)
    tb_logger.log_value('recall@5_text', r5, step=model.Eiters)
    tb_logger.log_value('recall@10_text', r10, step=model.Eiters)
    tb_logger.log_value('recall@1_im', r1i, step=model.Eiters)
    tb_logger.log_value('recall@5_im', r5i, step=model.Eiters)
    tb_logger.log_value('recall@10_im', r10i, step=model.Eiters)

    return currscore, sims
def encode_data(model, data_loader, log_step=10):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # np array to keep all the embeddings
    img_embs = None
    cap_embs = None
    cap_lens = None

    # pdb.set_trace()
    max_n_word = 0
    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # print(lengths)
        max_n_word = max(max_n_word, max(lengths))

    for i, (images, captions, lengths, ids) in enumerate(data_loader):
        # make sure val logger is used
        model.logger = val_logger

        # compute the embeddings
        # pdb.set_trace()
        img_emb, cap_emb,_,_ = model.forward_emb_obj(images, captions, volatile=True)
        #print(img_emb)
        if img_embs is None:
            if img_emb.dim() == 3:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1), img_emb.size(2)))
            else:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            cap_embs = np.zeros((len(data_loader.dataset), max_n_word, cap_emb.size(2)))
            cap_lens = [0] * len(data_loader.dataset)
        # cache embeddings
        img_embs[ids] = img_emb.data.cpu().numpy().copy()
        cap_embs[ids,:max(lengths),:] = cap_emb.data.cpu().numpy().copy()
        for j, nid in enumerate(ids):
            cap_lens[nid] = lengths[j]
        '''
        # measure accuracy and record loss
        model.forward_loss(img_emb, cap_emb, cap_len)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % log_step == 0:
            logging('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        '''
        del images, captions
    return img_embs, cap_embs, cap_lens

def encode_data_bak(model, data_loader, opt):
    """Encode all images and captions loadable by `data_loader`
    """
    batch_time = AverageMeter()
    val_logger = LogCollector()

    # switch to evaluate mode
    model.val_start()

    end = time.time()

    # numpy array to keep all the embeddings
    img_embs_sub = []
    img_embs_obj = []
    img_embs_phr = []
    cap_embs_sub = []
    cap_embs_obj = []
    cap_embs_phr = []
    cap_embs_objects = []

    img_cls_sub = []
    img_cls_obj = []
    img_cls_phr = []
    img_embs_objects = []
    cap_cls_sub = []
    cap_cls_obj = []
    cap_cls_phr = []
    cap_cls_objects = []
    for i, test_data in enumerate(data_loader):
    # for i, subFea,objFea,preFea,subject_ind,object_ind,predicate_ind,subject_boxe,object_boxe,im_info, \
    #     subTxt,objTxt,preTxt,objectsTxt,ids in enumerate(data_loader):

        subFea,objFea,preFea,subject_ind,object_ind,predicate_ind,subject_boxe,object_boxe,im_info, \
        subTxt,objTxt,preTxt,objectsTxt,ids = test_data[:]

        
        images_sub = subFea.view(-1, opt.img_dim) 
        images_obj = objFea.view(-1, opt.img_dim) 
        images_phr = preFea.view(-1, opt.img_dim) 

        imageOnly = False
        # captions_obj = torch.cat((subTxt, objTxt), 0)
        if len(subTxt.size()) < 3:
            # pdb.set_trace()
            captions_sub = None
            captions_obj = None
            captions_phr = None
            imageOnly = True
        else:
            captions_sub = subTxt.view(-1, opt.text_dim) 
            captions_obj = objTxt.view(-1, opt.text_dim) 
            captions_phr = preTxt.view(-1, opt.text_dim) 

        captions_objects = objectsTxt.view(-1, opt.text_dim) 

        # make sure val logger is used
        model.logger = val_logger
        
        # compute the embeddings
        # img_emb, cap_emb, kld_outs = model.forward_emb(images, captions, lengths, volatile=True)
        img_emb_sub, cap_emb_sub, img_cls_score_sub, txt_sub_cls_score_sub = model.forward_emb_obj(images_sub, captions_sub, imageOnly = imageOnly)
        img_emb_obj, cap_emb_obj, img_obj_cls_score_obj, txt_obj_cls_score_obj = model.forward_emb_obj(images_obj, captions_obj, imageOnly = imageOnly)
        img_emb_phr, cap_emb_phr, img_cls_score_predicate, txt_cls_score_predicate = model.forward_emb_phr(images_phr, captions_phr, imageOnly = imageOnly)
        _, cap_emb_objects, _, txt_cls_score_objects = model.forward_emb_obj(None, captions_objects, captionOnly = True)

        # pre_scores_img_obj, pre_inds = cls_prob_predicate[:, 1:].data.max(1)
        # predicate_inds += 1
        if cap_emb_sub is not None:
            pre_scores_cap_sub, pre_inds_cap_sub = txt_sub_cls_score_sub.data.max(1)
            pre_scores_cap_obj, pre_inds_cap_obj = txt_obj_cls_score_obj.data.max(1)
            pre_scores_cap_phr, pre_inds_cap_phr = txt_cls_score_predicate.data.max(1)

        pre_scores_img_sub, pre_inds_img_sub = img_cls_score_sub.data.max(1)
        pre_scores_img_obj, pre_inds_img_obj = img_obj_cls_score_obj.data.max(1)
        pre_scores_img_phr, pre_inds_img_phr = img_cls_score_predicate.data.max(1)

        pre_scores_cap_objects, pre_inds_cap_objects = txt_cls_score_objects.data.max(1)

        # initialize the numpy arrays given the size of the embeddings
        # if img_embs_sub is None:
            # img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
            # cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))

            # img_embs_sub = img_emb_sub.data.cpu().numpy().copy()
            # img_embs_obj = img_emb_obj.data.cpu().numpy().copy()
            # img_embs_phr = img_emb_phr.data.cpu().numpy().copy()
            # cap_embs_sub = cap_emb_sub.data.cpu().numpy().copy()
            # cap_embs_obj = cap_emb_obj.data.cpu().numpy().copy()
            # cap_embs_phr = cap_emb_phr.data.cpu().numpy().copy()
            # cap_embs_objects = cap_emb_objects.data.cpu().numpy().copy()

            # img_cls_sub = pre_inds_img_sub.cpu().numpy().copy()
            # img_cls_obj = pre_inds_img_obj.cpu().numpy().copy()
            # img_cls_phr = pre_inds_img_phr.cpu().numpy().copy()
            # cap_cls_sub = pre_inds_cap_sub.cpu().numpy().copy()
            # cap_cls_obj = pre_inds_cap_obj.cpu().numpy().copy()
            # cap_cls_phr = pre_inds_cap_phr.cpu().numpy().copy()
            # cap_cls_objects = pre_inds_cap_objects.cpu().numpy().copy()

        # preserve the embeddings by copying from gpu and converting to numpy
        # img_embs[ids] = img_emb.data.cpu().numpy().copy()
        # cap_embs[ids] = cap_emb.data.cpu().numpy().copy()
        # else:
        if cap_emb_sub is None:
            cap_embs_sub.append(0)
            cap_embs_obj.append(0)
            cap_embs_phr.append(0)
            cap_cls_sub.append(0)
            cap_cls_obj.append(0)
            cap_cls_phr.append(0)

        else:
            cap_embs_sub.append(cap_emb_sub.data.cpu().numpy().copy())
            cap_embs_obj.append(cap_emb_obj.data.cpu().numpy().copy())
            cap_embs_phr.append(cap_emb_phr.data.cpu().numpy().copy())
            cap_cls_sub.append(pre_inds_cap_sub.cpu().numpy().copy())
            cap_cls_obj.append(pre_inds_cap_obj.cpu().numpy().copy())
            cap_cls_phr.append(pre_inds_cap_phr.cpu().numpy().copy())


        img_embs_sub.append(img_emb_sub.data.cpu().numpy().copy())
        img_embs_obj.append(img_emb_obj.data.cpu().numpy().copy())
        img_embs_phr.append(img_emb_phr.data.cpu().numpy().copy())

        img_emb_objects = np.concatenate((img_emb_sub.data.cpu().numpy().copy(),img_emb_obj.data.cpu().numpy().copy()),axis = 0)
        img_embs_objects = append(img_emb_objects)

        img_cls_sub.append(pre_inds_img_sub.cpu().numpy().copy())
        img_cls_obj.append(pre_inds_img_obj.cpu().numpy().copy())
        img_cls_phr.append(pre_inds_img_phr.cpu().numpy().copy())

        cap_embs_objects.append(cap_emb_objects.data.cpu().numpy().copy())
        cap_cls_objects.append(pre_inds_cap_objects.cpu().numpy().copy())

        # measure accuracy and record loss
        # model.forward_loss(img_emb, cap_emb, lengths, captions, kld_outs)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % opt.log_step == 0:
            print('Test: [{0}/{1}]\t'
                    '{e_log}\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    .format(
                        i, len(data_loader), batch_time=batch_time,
                        e_log=str(model.logger)))
        del subFea,objFea,preFea,subject_ind,object_ind,predicate_ind,subject_boxe,object_boxe,im_info,subTxt,objTxt,preTxt,objectsTxt

    return (img_embs_sub, img_embs_obj, img_embs_phr, cap_embs_sub, cap_embs_obj, cap_embs_phr, cap_embs_objects), \
            (img_cls_sub, img_cls_obj, img_cls_phr, cap_cls_sub, cap_cls_obj, cap_cls_phr, cap_cls_objects)

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
