import os
import pickle
import numpy
import time
import numpy as np
import torch
import pdb
from collections import OrderedDict
import sys
from torch.autograd import Variable
from model import xattn_score_t2i, xattn_score_i2t
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)

class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.iteritems()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.iteritems():
            tb_logger.log_value(prefix + k, v.val, step=step)        

def cosine_sim(im, s):
    """Cosine similarity between all the image and sentence pairs
    """
    return im.mm(s.t())


def order_sim(im, s):
    """Order embeddings similarity measure $max(0, s-im)$
    """
    YmX = (s.unsqueeze(1).expand(s.size(0), im.size(0), s.size(1))
           - im.unsqueeze(0).expand(s.size(0), im.size(0), s.size(1)))
    # score = -YmX.clamp(min=0).pow(2).sum(2).squeeze(2).sqrt().t()
    score = -YmX.clamp(min=0).pow(2).sum(2).sqrt().t()
    return score

def label_sim(img_l_s,cap_l_s,img_l_o,cap_l_o,img_l_r,cap_l_r,cap_l_obj):
    """
    one img, one cap
    200*1,2n*1,100*1,n*1
    """ 
    if isinstance(cap_l_s,int):
        img_l_s = set(img_l_s.tolist())
        img_l_o = set(img_l_o.tolist())
        cap_l_obj = set(cap_l_obj.tolist())  

        img_l = img_l_s.union(img_l_o)
             
        common_o = cap_l_obj.intersection(img_l)
        if 0 in common_o:
            common_o.remove(0)
        socre_o = len(common_o)
        # socre_r = 0
        common_RDF = []

    else:
        img_l_s = img_l_s.tolist()
        cap_l_s = cap_l_s.tolist()
        img_l_o = img_l_o.tolist()
        cap_l_o = cap_l_o.tolist()
        img_l_r = img_l_r.tolist()
        cap_l_r = cap_l_r.tolist()
        cap_l_obj = cap_l_obj.tolist()

        img_l = set(img_l_s).union(set(img_l_o))


        common_o = set(cap_l_obj).intersection(img_l)
        # common_r = set(img_l_r).intersection(set(cap_l_r))
        if 0 in common_o:
            common_o.remove(0)

        socre_o = len(common_o)
        # socre_r = len(common_r)

        img_RDF = []
        cap_RDF = []
        for i in range(len(img_l_s)):
            if img_l_s[i] !=0 and img_l_r[i] != 0 and img_l_o[i] != 0:
                img_RDF.append(str(img_l_s[i])+'_'+str(img_l_r[i])+'_'+str(img_l_o[i]))

        for i in range(len(cap_l_s)):
            if cap_l_s[i] !=0 and cap_l_r[i] != 0 and cap_l_o[i] != 0:
                cap_RDF.append(str(cap_l_s[i])+'_'+str(cap_l_r[i])+'_'+str(cap_l_o[i]))

        img_RDF = set(img_RDF)
        cap_RDF = set(cap_RDF)

        common_RDF = img_RDF.intersection(cap_RDF)
    # if(0 in cap_l_obj and 0 in img_l):
    #     pdb.set_trace()
    # return float(socre_o+socre_r+len(common_RDF))
    return float(socre_o+len(common_RDF))

def label_sim_bak(img_l_s,cap_l_s,img_l_o,cap_l_o,img_l_r,cap_l_r,cap_l_obj):
    """
    one img, one cap
    200*1,2n*1,100*1,n*1
    """ 
    if isinstance(cap_l_s,int):
        img_l_s = img_l_s.tolist()
        img_l_o = img_l_o.tolist()
        cap_l_obj = cap_l_obj.tolist()  
             
        common_o = [x for x in img_l_s+img_l_o if x in cap_l_obj]
        socre_o = len(common_o)/len(cap_l_obj)
        socre_r = 0
        common_RDF = []

    else:
        img_l_s = img_l_s.tolist()
        cap_l_s = cap_l_s.tolist()
        img_l_o = img_l_o.tolist()
        cap_l_o = cap_l_o.tolist()
        img_l_r = img_l_r.tolist()
        cap_l_r = cap_l_r.tolist()
        cap_l_obj = cap_l_obj.tolist()

        common_o = [x for x in img_l_s+img_l_o if x in cap_l_obj]
        common_r = [x for x in img_l_r if x in cap_l_r]
        socre_o = len(common_o)/len(cap_l_obj)
        socre_r = len(common_r)/len(cap_l_r)
        
        img_RDF = []
        cap_RDF = []
        for i in range(len(img_l_s)):
            img_RDF.append(str(img_l_s[i])+'_'+str(img_l_r[i])+'_'+str(img_l_o[i]))

        for i in range(len(cap_l_s)):
            cap_RDF.append(str(cap_l_s[i])+'_'+str(cap_l_r[i])+'_'+str(cap_l_o[i]))

        common_RDF = [x for x in img_RDF if x in cap_RDF]

    # return float(socre_o+socre_r+len(common_RDF))
    return float(socre_o)

def multi_fea_sim(img_f_s,cap_f_s,img_f_o,cap_f_o,img_f_r,cap_f_r,cap_f_obj):
    """
    one img, one cap
    200*1024,2n*1,100*1,n*1
    """ 
    common_o = [x for x in [img_l_s+img_l_o] if x in cap_l_obj]
    common_r = [x for x in img_l_r if x in cap_l_r]
    socre_o = num_multi_fea_sim([img_l_s+img_l_o],cap_l_obj)/cap_l_obj.shape[0]
    socre_r = num_multi_fea_sim(img_l_r,cap_l_r)/cap_l_r.shape[0]

    score_RDF = numR_multi_fea_sim(img_f_s,img_f_o,img_f_r,cap_f_s,cap_f_o,cap_f_r)

    return socre_o+socre_r+score_RDF

def num_multi_fea_sim(x, y, thr = 0.5):
    """
    x, y
    a*1024,b*1024
    """ 
    sim = cosine_sim(x, y)
    simS = sim.max(axis=1)
    simS_inds = np.where(simS >= thr)[0]

    return len(simS_inds)

def numR_multi_fea_sim(x, y, r, x1, y1, r1, thr = 0.5):
    """
    x, y
    a*1024,b*1024
    """ 
    socre = 0
    simX = cosine_sim(x, x1)
    simY = cosine_sim(y, y1)
    simR = cosine_sim(r, r1)

    for i in range(x1.shape[0]):
        for j in range(x.shape[0]):
            if(simX[j,i]>=thr and simY[j,i]>=thr and simR[j,i]>=thr):
                socre+=1

    return socre


def re_test_lab(img_cls_sub, cap_cls_sub, img_cls_obj, cap_cls_obj, img_cls_phr, cap_cls_phr, cap_cls_objects, topN = 100, measure='cosine'):
    """
    Images->Text (Image Annotation)
    Images: (5N, K) matrix of images
    Captions: (5N, K) matrix of captions
    """
    # pdb.set_trace()
    npts = len(img_cls_sub)
    npts_img = npts/5

    scores = numpy.zeros((npts_img,npts),dtype = 'float')

    for i in range(npts_img):
        if i % 100 == 0:
            print(i)
        for j in range(npts):
            # Get query image
            # im = images[i].reshape(1, topN, -1)
            # cp = captions[j].reshape(1, topN, -1)
            if isinstance(cap_cls_sub[j],int):
                cap_l_s = 0
                cap_l_o = 0
                cap_l_r = 0               
            else:
                cap_l_s = cap_cls_sub[j]
                cap_l_o = cap_cls_obj[j]
                cap_l_r = cap_cls_phr[j]

            img_l_s = img_cls_sub[i*5]
            img_l_o = img_cls_obj[i*5]
            img_l_r = img_cls_phr[i*5]
            cap_l_obj = cap_cls_objects[j]

            # Compute scores
            curScore = label_sim(img_l_s,cap_l_s,img_l_o,cap_l_o,img_l_r,cap_l_r,cap_l_obj)
            # print(curScore)
            scores[i][j] = curScore
    
    ranks_i2t = []
    rankind_i2t = (-scores).argsort(axis=1)
    for i in range(npts_img):        
        record = 1e20
        for j in range(10):
            if((i+1)*5>rankind_i2t[i][j] and i*5<=rankind_i2t[i][j]):
                record = j
                break
        ranks_i2t.append(record)

    ranks_t2i = []
    rankind_t2i = (-scores).argsort(axis=0)
    for i in range(npts):
        record = 1e20
        for j in range(10):#(res(ii,j)*5)>=ii&&((res(ii,j)-1)*5)<ii
            if(i>=rankind_t2i[j][i]*5 and i<(rankind_t2i[j][i]+1)*5):
                record = j
                break
        ranks_t2i.append(record)
  
    ranks_i2t = np.array(ranks_i2t)
    ranks_t2i = np.array(ranks_t2i)
    # Compute metrics
    i2t_r1 = 100.0 * len(numpy.where(ranks_i2t < 1)[0]) / len(ranks_i2t)
    i2t_r5 = 100.0 * len(numpy.where(ranks_i2t < 5)[0]) / len(ranks_i2t)
    i2t_r10 = 100.0 * len(numpy.where(ranks_i2t < 10)[0]) / len(ranks_i2t)

    t2i_r1 = 100.0 * len(numpy.where(ranks_t2i < 1)[0]) / len(ranks_t2i)
    t2i_r5 = 100.0 * len(numpy.where(ranks_t2i < 5)[0]) / len(ranks_t2i)
    t2i_r10 = 100.0 * len(numpy.where(ranks_t2i < 10)[0]) / len(ranks_t2i)
    # pdb.set_trace()
    return (i2t_r1, i2t_r5, i2t_r10, t2i_r1, t2i_r5, t2i_r10)

def i2t(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, captions, caplens, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)

def shard_xattn_t2i(images, captions, caplens, opt, shard_size=128):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = (len(images)-1)/shard_size + 1
    n_cap_shard = (len(captions)-1)/shard_size + 1
    
    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn_t2i batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).cuda()
            s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
            l = caplens[cap_start:cap_end]
            sim = xattn_score_t2i(im, s, l, opt)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d


def shard_xattn_i2t(images, captions, caplens, opt, shard_size=128):
    """
    Computer pairwise i2t image-caption distance with locality sharding
    """
    n_im_shard = (len(images)-1)/shard_size + 1
    n_cap_shard = (len(captions)-1)/shard_size + 1
    
    d = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size*i, min(shard_size*(i+1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn_i2t batch (%d,%d)' % (i,j))
            cap_start, cap_end = shard_size*j, min(shard_size*(j+1), len(captions))
            im = Variable(torch.from_numpy(images[im_start:im_end]), volatile=True).cuda()
            s = Variable(torch.from_numpy(captions[cap_start:cap_end]), volatile=True).cuda()
            l = caplens[cap_start:cap_end]
            sim = xattn_score_i2t(im, s, l, opt)
            d[im_start:im_end, cap_start:cap_end] = sim.data.cpu().numpy()
    sys.stdout.write('\n')
    return d