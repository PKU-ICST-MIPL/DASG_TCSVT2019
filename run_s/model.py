# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.init
import torchvision.models as models
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm
import numpy as np
from collections import OrderedDict
from faster_rcnn.utils.cython_bbox import bbox_overlaps, bbox_intersections
import json
import pdb
def read_cate_txt(filepath):
    txt_file = open(filepath,'r')
    ret = []

    while True:  
        line = txt_file.readline().strip()
        if line : 
            ret.append(line)
            
        else:
            break
    txt_file.close()
    return ret
def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()
def func_attention(query, context, opt, smooth, eps=1e-8):
    """
    query: (n_context, queryL, d)
    context: (n_context, sourceL, d)
    """
    batch_size_q, queryL = query.size(0), query.size(1)
    batch_size, sourceL = context.size(0), context.size(1)


    # Get attention
    # --> (batch, d, queryL)
    queryT = torch.transpose(query, 1, 2)

    # (batch, sourceL, d)(batch, d, queryL)
    # --> (batch, sourceL, queryL)
    attn = torch.bmm(context, queryT)
    if opt.raw_feature_norm == "softmax":
        # --> (batch*sourceL, queryL)
        attn = attn.view(batch_size*sourceL, queryL)
        attn = nn.Softmax()(attn)
        # --> (batch, sourceL, queryL)
        attn = attn.view(batch_size, sourceL, queryL)
    elif opt.raw_feature_norm == "l2norm":
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "clipped_l2norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l2norm(attn, 2)
    elif opt.raw_feature_norm == "l1norm":
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped_l1norm":
        attn = nn.LeakyReLU(0.1)(attn)
        attn = l1norm_d(attn, 2)
    elif opt.raw_feature_norm == "clipped":
        attn = nn.LeakyReLU(0.1)(attn)
    elif opt.raw_feature_norm == "no_norm":
        pass
    else:
        raise ValueError("unknown first norm type:", opt.raw_feature_norm)
    # --> (batch, queryL, sourceL)
    attn = torch.transpose(attn, 1, 2).contiguous()
    # --> (batch*queryL, sourceL)
    attn = attn.view(batch_size*queryL, sourceL)
    attn = nn.Softmax()(attn*smooth)
    # --> (batch, queryL, sourceL)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> (batch, sourceL, queryL)
    attnT = torch.transpose(attn, 1, 2).contiguous()

    # --> (batch, d, sourceL)
    contextT = torch.transpose(context, 1, 2)
    # (batch x d x sourceL)(batch x sourceL x queryL)
    # --> (batch, d, queryL)
    weightedContext = torch.bmm(contextT, attnT)
    # --> (batch, queryL, d)
    weightedContext = torch.transpose(weightedContext, 1, 2)

    return weightedContext, attnT


def xattn_score_t2i(images, captions, cap_lens, opt):
    """
    Images: (n_image, n_regions, d) matrix of images
    Captions: (n_caption, max_n_word, d) matrix of captions
    CapLens: (n_caption) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # --> (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_regions, d)
            weiContext: (n_image, n_word, d)
            attn: (n_image, n_region, n_word)
        """
        weiContext, attn = func_attention(cap_i_expand, images, opt, smooth=opt.lambda_softmax)
        cap_i_expand = cap_i_expand.contiguous()
        weiContext = weiContext.contiguous()
        # (n_image, n_word)
        row_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()       
            # print(row_sim.shape) 
            if len(row_sim.size()) > 1:
                row_sim = row_sim.sum(dim=1, keepdim=True)
            else:
                row_sim = row_sim.view([-1,1])
            row_sim = torch.log(row_sim)/opt.lambda_lse           
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    # pdb.set_trace()
    similarities = torch.cat(similarities, 1)

    
    return similarities


def xattn_score_i2t(images, captions, cap_lens, opt):
    """
    Images: (batch_size, n_regions, d) matrix of images
    Captions: (batch_size, max_n_words, d) matrix of captions
    CapLens: (batch_size) array of caption lengths
    """
    similarities = []
    n_image = images.size(0)
    n_caption = captions.size(0)
    n_region = images.size(1)
    for i in range(n_caption):
        # Get the i-th text description
        n_word = cap_lens[i]
        cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
        # (n_image, n_word, d)
        cap_i_expand = cap_i.repeat(n_image, 1, 1)
        """
            word(query): (n_image, n_word, d)
            image(context): (n_image, n_region, d)
            weiContext: (n_image, n_region, d)
            attn: (n_image, n_word, n_region)
        """
        weiContext, attn = func_attention(images, cap_i_expand, opt, smooth=opt.lambda_softmax)
        # (n_image, n_region)
        row_sim = cosine_similarity(images, weiContext, dim=2)
        if opt.agg_func == 'LogSumExp':
            row_sim.mul_(opt.lambda_lse).exp_()
            row_sim = row_sim.sum(dim=1, keepdim=True)
            row_sim = torch.log(row_sim)/opt.lambda_lse
        elif opt.agg_func == 'Max':
            row_sim = row_sim.max(dim=1, keepdim=True)[0]
        elif opt.agg_func == 'Sum':
            row_sim = row_sim.sum(dim=1, keepdim=True)
        elif opt.agg_func == 'Mean':
            row_sim = row_sim.mean(dim=1, keepdim=True)
        else:
            raise ValueError("unknown aggfunc: {}".format(opt.agg_func))
        similarities.append(row_sim)

    # (n_image, n_caption)
    similarities = torch.cat(similarities, 1)
    return similarities
class FC(nn.Module):
    def __init__(self, in_features, out_features, relu=True):
        super(FC, self).__init__()
        self.fc = nn.Linear(in_features, out_features)
        self.relu = nn.ReLU(inplace=False) if relu else None
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, x):
        x = self.fc(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
def l1norm(X, dim, eps=1e-8):
    """L1-normalize columns of X
    """
    norm = torch.abs(X).sum(dim=dim, keepdim=True) + eps
    X = torch.div(X, norm)
    return X


def l2norm(X, dim, eps=1e-8):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


class EncoderImage(nn.Module):

    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderImage, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        self.fc1 = nn.Linear(img_dim, 1024)
        self.fc2 = nn.Linear(1024, embed_size)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc1.in_features + self.fc1.out_features)
        self.fc1.weight.data.uniform_(-r, r)
        self.fc1.bias.data.fill_(0)

        r = np.sqrt(6.) / np.sqrt(self.fc2.in_features + self.fc2.out_features)
        self.fc2.weight.data.uniform_(-r, r)
        self.fc2.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc1(images)
        features = self.relu1(features)
        features = self.fc2(features)
        features_r = self.relu2(features)
        # normalize in the joint embedding space
        # if not self.no_imgnorm:
        #     features = l2norm(features)
        # take the absolute value of embedding (used in order embeddings)
        # if self.use_abs:
        #     features = torch.abs(features)
        return features, features_r

    # def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        # own_state = self.state_dict()
        # new_state = OrderedDict()
        # for name, param in state_dict.items():
        #     if name in own_state:
        #         new_state[name] = param

        # super(EncoderImage, self).load_state_dict(new_state)

class EncoderText(nn.Module):

    def __init__(self, txt_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderText, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        self.fc1 = nn.Linear(txt_dim, 1024)
        self.fc2 = nn.Linear(1024, embed_size)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc1.in_features + self.fc1.out_features)
        self.fc1.weight.data.uniform_(-r, r)
        self.fc1.bias.data.fill_(0)

        r = np.sqrt(6.) / np.sqrt(self.fc2.in_features + self.fc2.out_features)
        self.fc2.weight.data.uniform_(-r, r)
        self.fc2.bias.data.fill_(0)

    def forward(self, texts):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc1(texts)
        features = self.relu1(features)
        features = self.fc2(features)
        features_r = self.relu2(features)
        # normalize in the joint embedding space
        # if not self.no_imgnorm:
        #     features = l2norm(features)
        # take the absolute value of embedding (used in order embeddings)
        # if self.use_abs:
        #     features = torch.abs(features)
        return features,features_r

    # def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        # own_state = self.state_dict()
        # new_state = OrderedDict()
        # for name, param in state_dict.items():
        #     if name in own_state:
        #         new_state[name] = param

        # super(EncoderText, self).load_state_dict(new_state)

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

class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin=0, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

    def forward(self, im, s, s_l):
        # compute image-sentence score matrix
        if self.opt.cross_attn == 't2i':
            scores = xattn_score_t2i(im, s, s_l, self.opt)
        elif self.opt.cross_attn == 'i2t':
            scores = xattn_score_i2t(im, s, s_l, self.opt)
        else:
            raise ValueError("unknown first norm type:", opt.raw_feature_norm)
        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        mask = torch.eye(scores.size(0)) > .5
        I = Variable(mask)
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]
        return cost_s.sum() + cost_im.sum()



class UDAG(object):
    def __init__(self, opt):
        # Build Models
        self.opt = opt
        self.grad_clip = opt.grad_clip
        self.img_enc_obj = EncoderImage(opt.img_dim, opt.embed_size,use_abs=opt.use_abs,no_imgnorm=opt.no_imgnorm)
        self.txt_enc_obj = EncoderText(opt.text_dim, opt.embed_size,use_abs=opt.use_abs,no_imgnorm=opt.no_imgnorm)
        self.img_enc_phr = EncoderImage(opt.img_dim, opt.embed_size,use_abs=opt.use_abs,no_imgnorm=opt.no_imgnorm)
        self.txt_enc_phr = EncoderText(opt.text_dim, opt.embed_size,use_abs=opt.use_abs,no_imgnorm=opt.no_imgnorm)

        self.img_score_obj = FC(opt.embed_size, opt.n_classes_obj, relu=False)
        self.img_score_pred = FC(opt.embed_size, opt.n_classes_pred, relu=False)

        self.txt_score_obj = FC(opt.embed_size, opt.n_classes_obj, relu=False)
        self.txt_score_pred = FC(opt.embed_size, opt.n_classes_pred, relu=False)

        if torch.cuda.is_available():
            self.img_enc_obj.cuda()
            self.txt_enc_obj.cuda()
            self.img_enc_phr.cuda()
            self.txt_enc_phr.cuda()
            self.img_score_obj.cuda()
            self.img_score_pred.cuda()
            self.txt_score_obj.cuda()
            self.txt_score_pred.cuda()
            cudnn.benchmark = True
        # Loss and Optimizer
        self.criterion = ContrastiveLoss(opt=opt,
                                         margin=opt.margin,
                                         max_violation=opt.max_violation)
        params = list(self.img_enc_obj.parameters())
        params += list(self.txt_enc_obj.parameters())
        params += list(self.img_enc_phr.parameters())
        params += list(self.txt_enc_phr.parameters())
        params += list(self.img_score_obj.parameters())
        params += list(self.img_score_pred.parameters())
        params += list(self.txt_score_obj.parameters())
        params += list(self.txt_score_pred.parameters())
        # if opt.finetune:
        #     params += list(self.img_enc.cnn.parameters())
        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0
        self.gt_objects = np.load('/data2/chijingze/UDAG/vg/gt_objects.npy')
        self.gt_relationships = np.load('/data2/chijingze/UDAG/vg/gt_relationships.npy')
        self.gt_objects_cls = []
        self.gt_relationships_cls = []
        self.gt_objects_len = []
        self.gt_relationships_len = []
        for i in range(self.gt_objects.shape[0]):
            self.gt_objects_cls.append(self.gt_objects[i][:,4].astype(int))
            self.gt_objects_len.append(self.gt_objects_cls[i].shape[0])
        for i in range(self.gt_relationships.shape[0]):
            self.gt_relationships_cls.append(self.gt_relationships[i].flatten()[ np.flatnonzero(self.gt_relationships[i])])
            self.gt_relationships_len.append(self.gt_relationships_cls[i].shape[0])
        self.gt_objects_cls = np.array(self.gt_objects_cls)
        self.gt_relationships_cls = np.array(self.gt_relationships_cls)
        self.gt_objects_len = np.array(self.gt_objects_len)
        self.gt_relationships_len = np.array(self.gt_relationships_len)

        cats = json.load(open('/data2/chijingze/UDAG/vg/categories.json'))
        self._object_classes = tuple(['__background__'] + cats['object'])
        self._predicate_classes = tuple(['__background__'] + cats['predicate'])
        inverse_weight = json.load(open('/data2/chijingze/UDAG/vg/inverse_weight.json'))
        self.inverse_weight_object = torch.ones(opt.n_classes_obj)
        for idx in xrange(1, opt.n_classes_obj):
            self.inverse_weight_object[idx] = inverse_weight['object'][self._object_classes[idx]]
        self.inverse_weight_object = self.inverse_weight_object / self.inverse_weight_object.min()

        self.inverse_weight_predicate = torch.ones(opt.n_classes_pred)
        for idx in xrange(1, opt.n_classes_pred):
            self.inverse_weight_predicate[idx] = inverse_weight['predicate'][self._predicate_classes[idx]]
        self.inverse_weight_predicate = self.inverse_weight_predicate / self.inverse_weight_predicate.min()

        filepath = "/data1/chijingze/glove/glove.6B.300d.txt"
        file = open(filepath)
        self.gloveDic = {}
        while 1:
            line = file.readline().strip()
            if not line:
                break
            #print(line)
            words = line.split(' ')
            word = words[0]
            del (words[0])
            self.gloveDic[word] = words
        file.close()

        self.objDic = read_cate_txt('/data2/chijingze/UDAG/vg/object.txt')
        self.relDic = read_cate_txt('/data2/chijingze/UDAG/vg/relation.txt')


    def state_dict(self):
        state_dict = [self.img_enc_obj.state_dict(), self.txt_enc_obj.state_dict(),self.img_enc_phr.state_dict(), \
                     self.txt_enc_phr.state_dict(), self.img_score_obj.state_dict(),self.img_score_pred.state_dict(), \
                     self.txt_score_obj.state_dict(),self.txt_score_pred.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc_obj.load_state_dict(state_dict[0])
        self.txt_enc_obj.load_state_dict(state_dict[1])
        self.img_enc_phr.load_state_dict(state_dict[2])
        self.txt_enc_phr.load_state_dict(state_dict[3])
        self.img_score_obj.load_state_dict(state_dict[4])
        self.img_score_pred.load_state_dict(state_dict[5])
        self.txt_score_obj.load_state_dict(state_dict[6])
        self.txt_score_pred.load_state_dict(state_dict[7])

    def train_start(self):
        #switch to train mode
        self.img_enc_obj.train()
        self.txt_enc_obj.train()
        self.img_enc_phr.train()
        self.txt_enc_phr.train()
        self.img_score_obj.train()
        self.img_score_pred.train()
        self.txt_score_obj.train()
        self.txt_score_pred.train()

    def val_start(self):
        # switch to evaluate mode
        self.img_enc_obj.eval()
        self.txt_enc_obj.eval()
        self.img_enc_phr.eval()
        self.txt_enc_phr.eval()
        self.img_score_obj.eval()
        self.img_score_pred.eval()
        self.txt_score_obj.eval()
        self.txt_score_pred.eval()

    def forward_emb_obj(self, images, captions, volatile=False, captionOnly = False, imageOnly = False):
        """Compute the image and caption embeddings"""
        # Set mini-batch dataset
        if captionOnly and imageOnly:
            img_emb = None
            img_cls_score_object = None
            cap_emb = None
            txt_cls_score_object = None
        elif captionOnly and not imageOnly:
            captions = Variable(captions, volatile=volatile)
            if torch.cuda.is_available():
                captions = captions.cuda()
            # Forward
            cap_emb,cap_emb_r = self.txt_enc_obj(captions)
            txt_cls_score_object = self.txt_score_obj(cap_emb_r)
            img_emb = None
            img_cls_score_object = None

        elif imageOnly and not captionOnly:
            images = Variable(images, volatile=volatile)
            if torch.cuda.is_available():
                images = images.cuda()
            # Forward
            img_emb,img_emb_r = self.img_enc_obj(images)
            img_cls_score_object = self.img_score_obj(img_emb_r)
            cap_emb = None
            txt_cls_score_object = None
        else:
            images = Variable(images, volatile=volatile)
            captions = Variable(captions, volatile=volatile)
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()

            # Forward
            img_emb,img_emb_r = self.img_enc_obj(images)
            cap_emb,cap_emb_r = self.txt_enc_obj(captions)

            img_cls_score_object = self.img_score_obj(img_emb_r)
            txt_cls_score_object = self.txt_score_obj(cap_emb_r)
        return img_emb, cap_emb, img_cls_score_object, txt_cls_score_object

    def forward_emb_phr(self, images, captions, volatile=False, captionOnly = False,imageOnly = False):
        """Compute the image and caption embeddings"""
        # Set mini-batch dataset
        if imageOnly and captionOnly:
            cap_emb = None
            txt_cls_score_predicate = None
            img_emb = None
            img_cls_score_predicate = None

        elif imageOnly and not captionOnly:
            images = Variable(images, volatile=volatile)
            if torch.cuda.is_available():
                images = images.cuda()
            # Forward
            img_emb,img_emb_r = self.img_enc_phr(images)
            img_cls_score_predicate = self.img_score_pred(img_emb_r)
            cap_emb = None
            txt_cls_score_predicate = None
        elif captionOnly and not imageOnly:
            captions = Variable(captions, volatile=volatile)
            if torch.cuda.is_available():
                captions = captions.cuda()

            # Forward
            cap_emb,cap_emb_r = self.txt_enc_phr(captions)
            txt_cls_score_predicate = self.txt_score_pred(cap_emb_r)
            img_emb = None
            img_cls_score_predicate = None
        else:
            images = Variable(images, volatile=volatile)
            captions = Variable(captions, volatile=volatile)
            if torch.cuda.is_available():
                images = images.cuda()
                captions = captions.cuda()

            # Forward
            img_emb,img_emb_r = self.img_enc_phr(images)
            cap_emb,cap_emb_r = self.txt_enc_phr(captions)

            img_cls_score_predicate = self.img_score_pred(img_emb_r)
            txt_cls_score_predicate = self.txt_score_pred(cap_emb_r)
        return img_emb, cap_emb, img_cls_score_predicate, txt_cls_score_predicate

    def forward_loss(self, img_emb, cap_emb, cap_len):
        # Compute the loss given pairs of image and caption embeddings
        loss = self.criterion(img_emb, cap_emb, cap_len)
        self.logger.update('training loss', loss.data[0], img_emb.size(0))
        return loss

    def train_emb(self, *args):
        #subFea,objFea,preFea,subject_ind,object_ind,predicate_ind
        #subject_boxe,object_boxe,im_info, subTxt,objTxt,preTxt
        #One training step given images and captions.
        self.Eiters += 1
        self.logger.update('iterations', self.Eiters)
        self.logger.update('current learning rate', self.optimizer.param_groups[0]['lr'])

        subFea,objFea,preFea,subject_ind,object_ind,predicate_ind, \
        subject_boxe,object_boxe,im_info, subTxt,objTxt,preTxt,index = args[:]

        #image objects, bs*200*1024
        images_obj = torch.cat((subFea, objFea), 1) 
        #image phrases, bs*100*1024
        images_phr = preFea#.view(-1, self.opt.img_dim) 

        #text objects, bs*n*300
        captions_cls = self.gt_objects_cls[index] #bs*n
        
        captions_obj = self.ind_to_glove(captions_cls,self.objDic)#bs*n*300
        captions_obj = torch.Tensor(captions_obj)

        #text phr, bs*n*300
        captions_phr_cls = self.gt_relationships_cls[index] #bs*n
        captions_phr = self.ind_to_glove(captions_phr_cls,self.relDic)#bs*n*300
        captions_phr = torch.Tensor(captions_phr)

        # inds_obj = torch.cat((subject_ind, object_ind), 0)
        # inds_obj = inds_obj.view(-1, 1) 
        # inds_phr = predicate_ind.view(-1, 1) 

        # compute the embeddings
        img_emb_obj, cap_emb_obj, img_cls_score_object, txt_cls_score_object = self.forward_emb_obj(images_obj, captions_obj)
        img_emb_phr, cap_emb_phr, img_cls_score_predicate, txt_cls_score_predicate = self.forward_emb_phr(images_phr, captions_phr)

        obj_labels,phrase_labels = self.get_lable(self.gt_objects[index], self.gt_relationships[index],subject_boxe, object_boxe)  

        # fg_obj_labels = np.where(obj_labels > 0)[0]
        # fg_phrase_labels = np.where(phrase_labels > 0)[0]

        cap_gt_objects_len = self.gt_objects_len[index]
        cap_gt_relationships_len = self.gt_relationships_len[index]
        txt_cls_score_object_cut = None
        txt_cls_score_predicate_cut = None
        inds_obj = None
        inds_phr = None
        for i in range(cap_gt_objects_len.shape[0]):
            if txt_cls_score_object_cut is None:
                txt_cls_score_object_cut = txt_cls_score_object[i][:cap_gt_objects_len[i]]
                inds_obj = captions_cls[i]
            else:
                txt_cls_score_object_cut = torch.cat((txt_cls_score_object_cut, txt_cls_score_object[i][:cap_gt_objects_len[i]]), 0) 
                inds_obj = np.concatenate((inds_obj,captions_cls[i]),axis=0)

        for i in range(cap_gt_relationships_len.shape[0]):
            if txt_cls_score_predicate_cut is None:
                txt_cls_score_predicate_cut = txt_cls_score_predicate[i][:cap_gt_relationships_len[i]]
                inds_phr = captions_phr_cls[i]
            else:
                txt_cls_score_predicate_cut = torch.cat((txt_cls_score_predicate_cut, txt_cls_score_predicate[i][:cap_gt_relationships_len[i]]), 0)         
                inds_phr = np.concatenate((inds_phr,captions_phr_cls[i]),axis=0)

        obj_labels = Variable(torch.Tensor(obj_labels).long(), volatile=False)
        phrase_labels = Variable(torch.Tensor(phrase_labels).long(), volatile=False)
        inds_obj = Variable(torch.Tensor(inds_obj).long(), volatile=False)
        inds_phr = Variable(torch.Tensor(inds_phr).long(), volatile=False)

        if torch.cuda.is_available():
            obj_labels = obj_labels.cuda()
            phrase_labels = phrase_labels.cuda()
            inds_obj = inds_obj.cuda()
            inds_phr = inds_phr.cuda()

        self.optimizer.zero_grad()

        self.cross_entropy_object = self.build_loss_object(img_cls_score_object.view(-1, self.opt.n_classes_obj) , obj_labels) + \
                                    self.build_loss_object(txt_cls_score_object_cut, inds_obj)
        
        self.cross_entropy_predicate = self.build_loss_cls(img_cls_score_predicate.view(-1, self.opt.n_classes_pred) , phrase_labels) + \
                                        self.build_loss_cls(txt_cls_score_predicate_cut, inds_phr)
        # measure accuracy and record loss
        

        # pdb.set_trace()
        # lossCls = self.cross_entropy_object + self.cross_entropy_predicate


        lossConObj = self.forward_loss(img_emb_obj, cap_emb_obj, cap_gt_objects_len)
        lossConObj_show = lossConObj.data.cpu().numpy()[0]


        lossConPhr = self.forward_loss(img_emb_phr, cap_emb_phr, cap_gt_relationships_len)
        lossConPhr_show = lossConPhr.data.cpu().numpy()[0]

        if self.Eiters % self.opt.log_step == 0:
            # print('Epoch:[{0}]:{1},{2},{3}'.format(self.Eiters, lossCls.data.cpu().numpy()[0],lossConObj_show,lossConPhr_show ))
            print('Epoch:[{0}]:{1},{2},{3}'.format(self.Eiters, 0,lossConObj_show,lossConPhr_show ))
        


        # loss = lossCls+lossConObj+lossConPhr
        loss = lossConObj#+lossConPhr
        loss.backward()
        # compute gradient and do SGD step

        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()


    def build_loss_cls(self, cls_score, labels):
        labels = labels.squeeze()
        fg_cnt = torch.sum(labels.data.ne(0))
        bg_cnt = labels.data.numel() - fg_cnt

        ce_weights = np.sqrt(self.inverse_weight_predicate)
        ce_weights[0] = float(fg_cnt) / (bg_cnt + 1e-5)
        ce_weights = ce_weights.cuda()
        # print '[relationship]:'
        # print 'ce_weights:'
        # print ce_weights
        # print 'cls_score:'
        # print cls_score 
        # print 'labels'
        # print labels
        ce_weights = ce_weights.cuda()
        cross_entropy = nn.functional.cross_entropy(cls_score, labels, weight=ce_weights)

        # maxv, predict = cls_score.data.max(1)
        # if DEBUG:
        # print '[predicate]:'
        # if predict.sum() > 0:
        # print predict
        # print 'labels'
        # print labels

        # if fg_cnt == 0:
        #     tp = 0
        # else:
        #     tp = torch.sum(predict[bg_cnt:].eq(labels.data[bg_cnt:]))
        # tf = torch.sum(predict[:bg_cnt].eq(labels.data[:bg_cnt]))
        # fg_cnt = fg_cnt
        # bg_cnt = bg_cnt

        return cross_entropy#, tp, tf, fg_cnt, bg_cnt

    def build_loss_object(self, cls_score, roi_data):
        # classification loss
        # label = roi_data[1].squeeze()
        label = roi_data.squeeze()
        fg_cnt = torch.sum(label.data.ne(0)) #不等于0就返回1
        bg_cnt = label.data.numel() - fg_cnt

        ce_weights = np.sqrt(self.inverse_weight_object)
        ce_weights[0] = float(fg_cnt) / (bg_cnt + 1e-5)#调整第一个类别的权重，背景权重
        ce_weights = ce_weights.cuda()

        # maxv, predict = cls_score.data.max(1)
        # if fg_cnt > 0:
        #     self.tp = torch.sum(predict[:fg_cnt].eq(label.data[:fg_cnt]))
        # else:
        #     self.tp = 0.
        # if bg_cnt > 0:
        #     self.tf = torch.sum(predict[fg_cnt:].eq(label.data[fg_cnt:]))
        # else:
        #     self.tp = 0.
        # self.fg_cnt = fg_cnt
        # self.bg_cnt = bg_cnt

        # print '[object]:'
        # if predict.sum() > 0:
        # print predict

        # print 'accuracy: %2.2f%%' % (((self.tp + self.tf) / float(fg_cnt + bg_cnt)) * 100)
        # print predict
        cross_entropy = nn.functional.cross_entropy(cls_score, label, weight=ce_weights)
        # print cross_entropy

        return cross_entropy

    def get_lable(self, gt_objects, gt_relationships,subject_boxes, object_boxes):
        # rearrange the ground truth
        gt_n = gt_objects.shape[0]
        sub_labels_ret = None
        obj_labels_ret = None
        pair_labels_ret = None
        for i in range(gt_n):
            gt_object = gt_objects[i]
            gt_relationship = gt_relationships[i]
            subject_boxe = subject_boxes[i]
            object_boxe = object_boxes[i]

            sub_overlaps = bbox_overlaps(
                np.ascontiguousarray(subject_boxe, dtype=np.float),
                np.ascontiguousarray(gt_object[:, :4], dtype=np.float))
            obj_overlaps = bbox_overlaps(
                np.ascontiguousarray(object_boxe, dtype=np.float),
                np.ascontiguousarray(gt_object[:, :4], dtype=np.float))

            sub_gt_assignment = sub_overlaps.argmax(axis=1)
            sub_max_overlaps = sub_overlaps.max(axis=1)
            sub_labels = gt_object[sub_gt_assignment, 4]

            sub_fg_inds = np.where(sub_max_overlaps >= 0.5)[0]
            sub_bg_inds = np.where((sub_max_overlaps < 0.5) & (sub_max_overlaps >= 0.0))[0]
            sub_labels[sub_bg_inds] = 0

            obj_gt_assignment = obj_overlaps.argmax(axis=1)
            obj_max_overlaps = obj_overlaps.max(axis=1)
            obj_labels = gt_object[obj_gt_assignment, 4]

            obj_fg_inds = np.where(obj_max_overlaps >= 0.5)[0]
            obj_bg_inds = np.where((obj_max_overlaps < 0.5) & (obj_max_overlaps >= 0.0))[0]
            obj_labels[obj_bg_inds] = 0

            pair_labels = gt_relationship[sub_gt_assignment, obj_gt_assignment]
            keepPairsBG = np.where((sub_labels == 0) | (obj_labels == 0))[0]
            pair_labels[keepPairsBG] = 0

            if sub_labels_ret is None:
                sub_labels_ret = sub_labels
                obj_labels_ret = obj_labels
                pair_labels_ret = pair_labels
            else:
                sub_labels_ret = np.append(sub_labels_ret,sub_labels)
                obj_labels_ret = np.append(obj_labels_ret,obj_labels)
                pair_labels_ret = np.append(pair_labels_ret,pair_labels)

        obj_labels_all_ret = np.append(sub_labels_ret,obj_labels_ret)

        return obj_labels_all_ret,pair_labels_ret

    def get_lable_bak(self, gt_objects, gt_relationships,subject_boxes, object_boxes):
        # rearrange the ground truth
        gt_n = gt_objects.shape[0]
        sub_labels_ret = None
        obj_labels_ret = None
        pair_labels_ret = None
        for i in range(gt_n):
            gt_object = gt_objects[i]
            gt_relationship = gt_relationships[i]
            subject_boxe = subject_boxes[i]
            object_boxe = object_boxes[i]
            gt_rel_sub_idx, gt_rel_obj_idx = np.where(gt_relationship > 0) # ground truth number
            gt_sub = gt_object[gt_rel_sub_idx, :5]
            gt_obj = gt_object[gt_rel_obj_idx, :5]
            gt_rel = gt_relationship[gt_rel_sub_idx, gt_rel_obj_idx]

            # compute the overlap
            sub_overlaps = bbox_overlaps(
                np.ascontiguousarray(subject_boxe, dtype=np.float),
                np.ascontiguousarray(gt_sub[:, :4], dtype=np.float))
            obj_overlaps = bbox_overlaps(
                np.ascontiguousarray(object_boxe, dtype=np.float),
                np.ascontiguousarray(gt_obj[:, :4], dtype=np.float))


            sub_gt_assignment = sub_overlaps.argmax(axis=1)
            sub_max_overlaps = sub_overlaps.max(axis=1)
            sub_labels = gt_object[sub_gt_assignment, 4]#????

            sub_fg_inds = np.where(sub_max_overlaps >= 0.5)[0]
            sub_bg_inds = np.where((sub_max_overlaps < 0.5) & (sub_max_overlaps >= 0.0))[0]
            sub_labels[sub_bg_inds] = 0

            obj_gt_assignment = obj_overlaps.argmax(axis=1)
            obj_max_overlaps = obj_overlaps.max(axis=1)
            obj_labels = gt_object[obj_gt_assignment, 4]

            obj_fg_inds = np.where(obj_max_overlaps >= 0.5)[0]
            obj_bg_inds = np.where((obj_max_overlaps < 0.5) & (obj_max_overlaps >= 0.0))[0]
            obj_labels[obj_bg_inds] = 0


            pair_labels = gt_relationship[sub_gt_assignment, obj_gt_assignment]
            keepPairsBG = np.where((sub_labels == 0) | (obj_labels == 0))[0]
            pair_labels[keepPairsBG] = 0

            if sub_labels_ret is None:
                sub_labels_ret = sub_labels
                obj_labels_ret = obj_labels
                pair_labels_ret = pair_labels
            else:
                sub_labels_ret = np.append(sub_labels_ret,sub_labels)
                obj_labels_ret = np.append(obj_labels_ret,obj_labels)
                pair_labels_ret = np.append(pair_labels_ret,pair_labels)

            obj_labels_all_ret = np.append(sub_labels_ret,obj_labels_ret)

        return obj_labels_all_ret,pair_labels_ret
    def ind_to_glove(self,input,cate):
        ret = []
        maxLen = 0
        for i in range(input.shape[0]):
            maxLen = max(maxLen, input[i].shape[0])

        for i in range(input.shape[0]):
            input_i = input[i]
            feaTmp = []
            for j in range(input_i.shape[0]):
                feaTmp.append(self.getGlove(str(cate[input_i[j]]),'_'))
            for j in range(maxLen - input_i.shape[0]):
                feaTmp.append(np.zeros(300))
            ret.append(feaTmp)

        return np.array(ret)

    def getGlove(self,line,se):
        try:
            if se in line:
                classes = line.split(se)
                curVec = self.gloveDic[classes[0]]
                curVec = map(float, curVec)  
                curVec = np.array(curVec)

                for i in range(1,len(classes)):
                    tmpVec = self.gloveDic[classes[i]]
                    tmpVec = map(float, tmpVec)  
                    tmpVec = np.array(tmpVec)
                    curVec = curVec + tmpVec

                curVec = curVec/len(classes)    

                return curVec

            else:
                curVec = self.gloveDic[line]
                curVec = map(float, curVec)  
                return np.array(curVec)
        except Exception as e:
            print('Error:',e)    