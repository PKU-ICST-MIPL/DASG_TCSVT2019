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

import pdb
def l2norm(X):
    """L2-normalize columns of X
    """
    norm = torch.pow(X, 2).sum(dim=1).sqrt()
    #X = torch.div(X, norm.expand_as(X))
    X = torch.div(X, norm.unsqueeze(1).expand_as(X))
    return X


def EncoderImage(data_name, img_dim, embed_size, finetune=False,
                 cnn_type='vgg19', use_abs=False, no_imgnorm=False):
    """A wrapper to image encoders. Chooses between an encoder that uses
    precomputed image features, `EncoderImagePrecomp`, or an encoder that
    computes image features on the fly `EncoderImageFull`.
    """
    if data_name.endswith('_precomp'):
        img_enc = EncoderImagePrecomp(img_dim, embed_size, use_abs, no_imgnorm)
    else:
        img_enc = EncoderImageFull(embed_size, finetune, cnn_type, use_abs, no_imgnorm)

    return img_enc


# tutorials/09 - Image Captioning
class EncoderImageFull(nn.Module):

    def __init__(self, embed_size, finetune=False, cnn_type='vgg19',
                 use_abs=False, no_imgnorm=False):
        """Load pretrained VGG19 and replace top fc layer."""
        super(EncoderImageFull, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs

        # Load a pre-trained model
        self.cnn = self.get_cnn(cnn_type, True)

        # For efficient memory usage.
        for param in self.cnn.parameters():
            param.requires_grad = finetune

        # Replace the last fully connected layer of CNN with a new one
        if cnn_type.startswith('vgg'):
            self.fc = nn.Linear(self.cnn.classifier._modules['6'].in_features, embed_size)
            self.cnn.classifier = nn.Sequential(*list(self.cnn.classifier.children())[:-1])
        elif cnn_type.startswith('resnet'):
            self.fc = nn.Linear(self.cnn.module.fc.in_features, embed_size)
            self.cnn.module.fc = nn.Sequential()

        self.init_weights()

    def get_cnn(self, arch, pretrained):
        """Load a pretrained CNN and parallelize over GPUs
        """
        if pretrained:
            print("=> using pre-trained model '{}'".format(arch))
            model = models.__dict__[arch](pretrained=True)
        else:
            print("=> creating model '{}'".format(arch))
            model = models.__dict__[arch]()

        if arch.startswith('alexnet') or arch.startswith('vgg'):
            model.features = nn.DataParallel(model.features)
            model.cuda()
        else:
            model = nn.DataParallel(model).cuda()

        return model

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        features = self.cnn(images)
        # normalization in the image embedding space
        features = l2norm(features)
        # linear projection to the joint embedding space
        features = self.fc(features)
        # normalization in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)
        # take the absolute value of the embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)
        return features

class EncoderImagePrecomp(nn.Module):

    def __init__(self, img_dim, embed_size, use_abs=False, no_imgnorm=False):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.use_abs = use_abs
        self.fc = nn.Linear(img_dim, embed_size)
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features + self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, images):
        """Extract image feature vectors."""
        # assuming that the precomputed features are already l2-normalized
        features = self.fc(images)
        # normalize in the joint embedding space
        if not self.no_imgnorm:
            features = l2norm(features)
        # take the absolute value of embedding (used in order embeddings)
        if self.use_abs:
            features = torch.abs(features)
        return features

    def load_state_dict(self, state_dict):
        """Copies parameters. overwritting the default one to
        accept state_dict from Full model
        """
        own_state = self.state_dict()
        new_state = OrderedDict()
        for name, param in state_dict.items():
            if name in own_state:
                new_state[name] = param

        super(EncoderImagePrecomp, self).load_state_dict(new_state)


# tutorials/08 - Language Model
# RNN Based Language Model
class EncoderText(nn.Module):

    def __init__(self, opt):
        super(EncoderText, self).__init__()
        self.use_abs = opt.use_abs
        self.rnn_type = getattr(opt, 'rnn_type', 'GRU')
        self.num_layers = opt.num_layers
        self.embed_size = opt.embed_size
        self.batch_size = opt.batch_size
        # word embedding
        self.embed = nn.Embedding(opt.vocab_size, opt.word_dim)
        # caption embedding
        if 'Bi' in self.rnn_type:
            self.rnn = nn.GRU(opt.word_dim, opt.embed_size, opt.num_layers, bias=False, batch_first=True, bidirectional=1)
            self.bi_out = nn.Linear(opt.embed_size * 2, opt.embed_size)
        else:
            self.rnn = nn.GRU(opt.word_dim, opt.embed_size, opt.num_layers, batch_first=True)

        self.init_weights()

    def init_weights(self):
        self.embed.weight.data.uniform_(-0.1, 0.1)
        if 'Bi' in self.rnn_type:
            """Xavier initialization for the fully connected layer
            """
            r = np.sqrt(6.) / np.sqrt(self.bi_out.in_features + self.bi_out.out_features)
            self.bi_out.weight.data.uniform_(-r, r)
            self.bi_out.bias.data.fill_(0)

    def init_hidden(self, bsz):
        return Variable(torch.zeros(self.num_layers * 2, bsz, self.embed_size)).cuda()

    def forward_GRU(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x = self.embed(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        I = torch.LongTensor(lengths).view(-1, 1, 1)
        I = Variable(I.expand(x.size(0), 1, self.embed_size)-1).cuda()
        out = torch.gather(padded[0], 1, I).squeeze(1)

        # normalization in the joint embedding space
        out = l2norm(out)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)
        return out

    def forward_BiGRU(self, x, lengths):
        outs = []
        # Lengths data is wrapped inside a Variable.
        x = self.embed(x)
        hiddens = self.init_hidden(x.data.size(0))
        emb = pack_padded_sequence(x, lengths, batch_first=True)
        outputs, hidden_t = self.rnn(emb, hiddens)
        # sum_out = True
        output_unpack = pad_packed_sequence(outputs, batch_first=True)
        hidden_o = output_unpack[0].sum(1)
        output_lengths = Variable(torch.from_numpy(np.asarray(lengths)).float().cuda(), requires_grad=False)
        torch.div(hidden_o, output_lengths.view(-1, 1))
        hidden_map = self.bi_out(hidden_o)
        # normalization in the joint embedding space
        out = l2norm(hidden_map)

        # take absolute value, used by order embeddings
        if self.use_abs:
            out = torch.abs(out)
        return out

    def forward(self, x, lengths):
        if 'Bi' in self.rnn_type:
            out = self.forward_BiGRU(x, lengths)
        else:
            out = self.forward_GRU(x, lengths)
        return out

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

    def __init__(self, margin=0, measure=False, max_violation=False):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        if measure == 'order':
            self.sim = order_sim
        else:
            self.sim = cosine_sim

        self.max_violation = max_violation

    '''
    For single-modal retrieval, emb1=query, emb2=data
    For cross-modal retrieval, emb1=query in source domain, emb2=data in target domain
    '''
    def forward(self, emb1, emb2):
        # compute image-sentence score matrix
        scores = self.sim(emb1, emb2)
        diagonal = scores.diag().view(emb1.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row image retrieval
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


class XRN(object):
    def __init__(self, opt):
        # Build Models
        self.opt = opt
        self.text_enc_init = getattr(opt, 'text_enc_init', 0)
        self.grad_clip = opt.grad_clip
        self.img_enc = EncoderImage(opt.data_name, opt.img_dim, opt.embed_size,
                                    opt.finetune, opt.cnn_type,
                                    use_abs=opt.use_abs,
                                    no_imgnorm=opt.no_imgnorm)
        self.txt_enc = EncoderText(opt)
        if torch.cuda.is_available():
            self.img_enc.cuda()
            self.txt_enc.cuda()
            cudnn.benchmark = True
        # Loss and Optimizer
        self.criterion = ContrastiveLoss(margin=opt.margin,
                                         measure=opt.measure,
                                         max_violation=opt.max_violation)
        params = list(self.txt_enc.parameters())
        params += list(self.img_enc.fc.parameters())
        if opt.finetune:
            params += list(self.img_enc.cnn.parameters())
        self.params = params
        self.optimizer = torch.optim.Adam(params, lr=opt.learning_rate)
        self.Eiters = 0

        self.weight_init()

    def weight_init(self):
        if self.text_enc_init:
            init_path = 'save/coco_resnet_restval/model_best.pth.tar'
            print("=> loading checkpoint '{}'".format(init_path))
            checkpoint = torch.load(init_path)
            self.txt_enc.load_state_dict(checkpoint['model'][1])

    def state_dict(self):
        state_dict = [self.img_enc.state_dict(), self.txt_enc.state_dict()]
        return state_dict

    def load_state_dict(self, state_dict):
        self.img_enc.load_state_dict(state_dict[0])
        self.txt_enc.load_state_dict(state_dict[1])

    def train_start(self):
        #switch to train mode
        self.img_enc.train()
        self.txt_enc.train()

    def val_start(self):
        # switch to evaluate mode
        self.img_enc.eval()
        self.txt_enc.eval()

    def forward_emb(self, images, captions, lengths, volatile=False):
        """Compute the image and caption embeddings"""
        # Set mini-batch dataset
        images = Variable(images, volatile=volatile)
        captions = Variable(captions, volatile=volatile)
        if torch.cuda.is_available():
            images = images.cuda()
            captions = captions.cuda()

        # Forward
        img_emb = self.img_enc(images)
        cap_emb = self.txt_enc(captions, lengths)
        return img_emb, cap_emb, None

    def forward_loss(self, img_emb, cap_emb, lengths, captions, kld_outs):
        # Compute the loss given pairs of image and caption embeddings
        loss = self.criterion(img_emb, cap_emb)
        self.logger.update('training loss', loss.data[0], img_emb.size(0))
        return loss

    def train_emb(self, images, captions, lengths, ids=None, *args):
        #One training step given images and captions.
        self.Eiters += 1
        self.logger.update('iterations', self.Eiters)
        self.logger.update('current learning rate', self.optimizer.param_groups[0]['lr'])

        # compute the embeddings
        img_emb, cap_emb, _ = self.forward_emb(images, captions, lengths)

        # measure accuracy and record loss
        self.optimizer.zero_grad()

        loss = self.forward_loss(img_emb, cap_emb, None, None, None)

        # compute gradient and do SGD step
        loss.backward()
        if self.grad_clip > 0:
            clip_grad_norm(self.params, self.grad_clip)
        self.optimizer.step()
