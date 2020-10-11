# -----------------------------------------------------------
# Stacked Cross Attention Network implementation based on 
# https://arxiv.org/abs/1803.08024.
# "Stacked Cross Attention for Image-Text Matching"
# Kuang-Huei Lee, Xi Chen, Gang Hua, Houdong Hu, Xiaodong He
#
# Writen by Kuang-Huei Lee, 2018
# ---------------------------------------------------------------
"""Data provider"""

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import os
import nltk
from PIL import Image
import numpy as np
import json as jsonmod
import pdb
import time
def getGlove(line,dict,se):
    try:
        if se in line:
            classes = line.split(se)
            curVec = dict[classes[0]]
            curVec = map(float, curVec)  
            curVec = np.array(curVec)

            for i in range(1,len(classes)):
                tmpVec = dict[classes[i]]
                tmpVec = map(float, tmpVec)  
                tmpVec = np.array(tmpVec)
                curVec = curVec + tmpVec

            curVec = curVec/len(classes)    

            return curVec

        else:
            curVec = dict[line]
            curVec = map(float, curVec)  
            return np.array(curVec)
    except Exception as e:
        print('Error:',e)


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

def read_RDF_txt(filepath,dict):
    txt_file = open(filepath,'r')
    subFea = []
    objFea = []
    preFea = []
    objects = []
    len_RDF = []
    len_objects = []
    # num = 0
    while True:  
        line = txt_file.readline().strip()
        if line : 
            # num += 1
            # print(num)

            if '/#' in line:#have a RDF
                
                lines = line.split('/#')

                objectsTex = lines[0].split('/')
                RDFTex = lines[1].split('#')
                cur_len_RDF = 0
                cur_len_objects = 0
                subFeaTmp = []
                preFeaTmp = []
                objFeaTmp = []
                objectsTmp = []
                for o in objectsTex:
                    objectsTmp.append(getGlove(str(o),dict,' '))
                    cur_len_objects += 1

                for r in RDFTex:
                    RDFs = r.split('+')
                    subFeaTmp.append(getGlove(str(RDFs[0]),dict,' '))
                    preFeaTmp.append(getGlove(str(RDFs[1]),dict,' '))
                    objFeaTmp.append(getGlove(str(RDFs[2]),dict,' '))
                    cur_len_RDF+=1


                subFea.append(subFeaTmp)
                preFea.append(preFeaTmp)
                objFea.append(objFeaTmp)
                objects.append(objectsTmp)

            else:
                lines = line.split('/')
                cur_len_RDF = 0
                cur_len_objects = 0
                objectsTmp = []
                for o in lines:
                    if o != '':
                        objectsTmp.append(getGlove(str(o),dict,' '))
                        cur_len_objects += 1
                objects.append(objectsTmp)

                subFeaTmp = [0]
                preFeaTmp = [0]
                objFeaTmp = [0]
                subFea.append(subFeaTmp)
                preFea.append(preFeaTmp)
                objFea.append(objFeaTmp)

            len_RDF.append(cur_len_RDF)
            len_objects.append(cur_len_objects)
            
        else:
            break
    txt_file.close()
    return np.array(subFea),np.array(preFea),np.array(objFea),np.array(objects),np.array(len_objects,dtype=np.int),np.array(len_RDF,dtype=np.int)

def ind_to_glove(input,cate,dict):
    ret = []

    for i in range(input.shape[0]):
        input_i = input[i]
        feaTmp = []
        for j in range(input_i.shape[0]):
            feaTmp.append(getGlove(str(cate[input_i[j]]),dict,'_'))

        ret.append(feaTmp)

    return np.array(ret)

class PrecompDatasetTest(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_name):

        loc = data_path + '/' + data_name + '/'
        print('val data load start')
        time_start=time.time()
        # Image features
        self.subFeas = np.load(loc+'subFea.npy')
        self.objFeas = np.load(loc+'objFea.npy')
        self.preFeas = np.load(loc+'preFea.npy')
        self.subject_inds = np.load(loc+'subject_inds.npy')
        self.object_inds = np.load(loc+'object_inds.npy')
        self.predicate_inds = np.load(loc+'predicate_inds.npy')
        self.subject_boxes = np.load(loc+'subject_boxes.npy')
        self.object_boxes = np.load(loc+'object_boxes.npy')
        self.im_infos = np.load(loc+'im_info.npy')

        # Captions
        filepath = "/data1/chijingze/glove/glove.6B.300d.txt"
        file = open(filepath)

        gloveDic = {}
        while 1:
            line = file.readline().strip()
            if not line:
                break
            #print(line)
            words = line.split(' ')
            word = words[0]
            del (words[0])
            gloveDic[word] = words
        file.close()
        # gloveDic = np.load('/data2/chijingze/UDAG/gloveDict.npy')
        self.subTxts,self.preTxts,self.objTxts,self.objectsTxts,self.len_objects,self.len_RDF = read_RDF_txt(loc+'test_caps_RDF.txt',gloveDic)
        
        self.length = self.subTxts.shape[0]

        time_elapsed = time.time() - time_start
        print('val data load successful {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        # pdb.set_trace()

    def __getitem__(self, index):
        imgIndex = index/5
        subFea = torch.Tensor(self.subFeas[imgIndex])
        objFea = torch.Tensor(self.objFeas[imgIndex])
        preFea = torch.Tensor(self.preFeas[imgIndex])
        subject_ind = torch.Tensor(self.subject_inds[imgIndex])
        object_ind = torch.Tensor(self.object_inds[imgIndex])
        predicate_ind = torch.Tensor(self.predicate_inds[imgIndex])
        subject_boxe = torch.Tensor(self.subject_boxes[imgIndex])
        object_boxe = torch.Tensor(self.object_boxes[imgIndex])
        im_info = torch.Tensor(self.im_infos[imgIndex])

        subTxt = torch.Tensor(self.subTxts[index])
        preTxt = torch.Tensor(self.preTxts[index])
        objTxt = torch.Tensor(self.objTxts[index])
        objectsTxt = torch.Tensor(self.objectsTxts[index])
        len_obj = self.len_objects[index]

        # return subFea,objFea,preFea,subject_ind,object_ind,predicate_ind,subject_boxe,object_boxe,im_info, \
        #         subTxt,objTxt,preTxt,objectsTxt,len_obj,index
        # imgObjFea = torch.cat(inputs=(subFea, objFea), dimension=0)
        imgObjFea = torch.cat([subFea, objFea], 0)

        return imgObjFea,objectsTxt,len_obj,index

    def __len__(self):
        return self.length

class PrecompDatasetTrain(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_name):

        loc = data_path + '/' + data_name + '/'
        print('train data load start')
        time_start=time.time()
        # Image features
        self.subFeas = np.load(loc+'subFea.npy')
        self.objFeas = np.load(loc+'objFea.npy')
        self.preFeas = np.load(loc+'preFea.npy')
        self.subject_inds = np.load(loc+'subject_inds.npy')
        self.object_inds = np.load(loc+'object_inds.npy')
        self.predicate_inds = np.load(loc+'predicate_inds.npy')
        self.subject_boxes = np.load(loc+'subject_boxes.npy')
        self.object_boxes = np.load(loc+'object_boxes.npy')
        self.im_infos = np.load(loc+'im_info.npy')


        # Captions
        filepath = "/data1/chijingze/glove/glove.6B.300d.txt"
        file = open(filepath)
        gloveDic = {}
        while 1:
            line = file.readline().strip()
            if not line:
                break
            #print(line)
            words = line.split(' ')
            word = words[0]
            del (words[0])
            gloveDic[word] = words
        file.close()
        objDic = read_cate_txt('/data2/chijingze/UDAG/vg/object.txt')
        relDic = read_cate_txt('/data2/chijingze/UDAG/vg/relation.txt')

        self.subTxts = ind_to_glove(self.subject_inds,objDic,gloveDic)
        self.preTxts = ind_to_glove(self.predicate_inds,relDic,gloveDic)
        self.objTxts = ind_to_glove(self.object_inds,objDic,gloveDic)
        
        self.length = self.subFeas.shape[0]

        time_elapsed = time.time() - time_start
        print('train data load successful {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        # pdb.set_trace()

    def __getitem__(self, index):
        subFea = torch.Tensor(self.subFeas[index])
        objFea = torch.Tensor(self.objFeas[index])
        preFea = torch.Tensor(self.preFeas[index])

        subject_ind = torch.Tensor(self.subject_inds[index])
        object_ind = torch.Tensor(self.object_inds[index])
        predicate_ind = torch.Tensor(self.predicate_inds[index])

        subject_boxe = torch.Tensor(self.subject_boxes[index])
        object_boxe = torch.Tensor(self.object_boxes[index])

        im_info = torch.Tensor(self.im_infos[index])
        # gt_object = torch.Tensor(self.gt_objects[index])
        # gt_relationship = torch.Tensor(self.gt_relationships[index])


        subTxt = torch.Tensor(self.subTxts[index])
        preTxt = torch.Tensor(self.preTxts[index])
        objTxt = torch.Tensor(self.objTxts[index])
        # objectsTxt = torch.Tensor(self.objectsTxts[index])

        # print(subFea.size())
        # print(objFea.size())
        # print(preFea.size())
        # print(subject_ind.size())
        # print(object_ind.size())
        # print(predicate_ind.size())
        # print(subject_boxe.size())
        # print(object_boxe.size())
        # print(im_info.size())
        # print(gt_object.size())
        # print(gt_relationship.size())
        # print(subTxt.size())
        # print(preTxt.size())
        # print(objTxt.size())
        return subFea,objFea,preFea,subject_ind,object_ind,predicate_ind, \
                subject_boxe,object_boxe,im_info, subTxt,objTxt,preTxt,index

    def __len__(self):
        return self.length

class PrecompDataset_ori(data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f30k_precomp, coco_precomp
    """

    def __init__(self, data_path, data_split, vocab):
        self.vocab = vocab
        loc = data_path + '/'

        # Captions
        self.captions = []
        with open(loc+'%s_caps.txt' % data_split, 'rb') as f:
            for line in f:
                self.captions.append(line.strip())

        # Image features
        self.images = np.load(loc+'%s_ims.npy' % data_split)
        self.length = len(self.captions)
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        if self.images.shape[0] != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000 

    def __getitem__(self, index):
        # handle the image redundancy
        img_id = index/self.im_div
        image = torch.Tensor(self.images[img_id])
        caption = self.captions[index]
        vocab = self.vocab

        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(
            str(caption).lower().decode('utf-8'))
        caption = []
        caption.append(vocab('<start>'))
        caption.extend([vocab(token) for token in tokens])
        caption.append(vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target, index, img_id

    def __len__(self):
        return self.length

def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids


def get_precomp_loader(data_path, data_split, opt, data_name,  batch_size=100,
                       shuffle=True, num_workers=2):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if data_split == 'train':
        dset = PrecompDatasetTrain(data_path, data_name)
    else:
        dset = PrecompDatasetTest(data_path, data_name)

    data_loader = torch.utils.data.DataLoader(dataset=dset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers)
                                              # collate_fn=collate_fn)
    return data_loader


def get_loaders(data_name, batch_size, workers, opt):

    train_loader = get_precomp_loader(opt.data_path, 'train', opt, 'vg',
                                      batch_size, True, workers)
    val_loader = get_precomp_loader(opt.data_path, 'dev', opt, data_name,
                                    1, False, workers)
    #yuchi temp
    # train_loader = None
    return train_loader, val_loader


def get_test_loader( data_name,  batch_size, workers, opt):
    val_loader = get_precomp_loader(opt.data_path, 'dev', opt, data_name,
                                    1, False, workers)
    return val_loader
