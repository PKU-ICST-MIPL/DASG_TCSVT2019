import skimage
import os
import pickle
from six.moves import cPickle
import numpy as np
import torch
import torch.nn as nn
import argparse
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from vocab import Vocabulary  # NOQA
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
# RL setting
parser.add_argument('--save_dir', type=str, default='./')
parser.add_argument('--vocab_path', default='./vocab/', help='Path to saved vocabulary pickle files.')
# Data input settings
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--dim', type=int, default=1024)
parser.add_argument('--dim_word', type=int, default=300)  # 300

parser.add_argument('--eval_batch_size', type=int, default=1000)
parser.add_argument('--validFreq', type=int, default=150)  # 300
parser.add_argument('--valid_num', type=int, default=1000)  # 300

# misc
parser.add_argument('--id', type=str, default='')
parser.add_argument('--multi_gpu', type=int, default=1)
parser.add_argument('--n_threads', type=int, default=1)
parser.add_argument('--n_gpus', type=int, default=1)
parser.add_argument('--gpus', default=[0, 1], nargs='+', type=int)
parser.add_argument('--cuda', type=int, default=1)

args = parser.parse_args()

def visualize_embeddings(font_size, save_dir, n_words, wemb, index2word, filter_words, word_inds=None):
    # Visualize embeddings
    # Pick some random words
    colors = cm.rainbow(np.linspace(0.2, 0.8, len(word_inds)))
    TEXT_KW = dict(fontsize=font_size, fontweight='normal')
    # Create embedding by summing left and right embeddings
    var_word_inds = Variable(torch.from_numpy(word_inds), volatile=False).cuda()
    w_embed = wemb(var_word_inds)
    # do PCA
    X = w_embed.data.cpu().numpy()
    pca = PCA(n_components=2)
    pca.fit(X)
    print pca.explained_variance_ratio_
    X = pca.transform(X)
    xs = X[:, 0]
    ys = X[:, 1]
    # draw
    plt.scatter(xs, ys, s=20, marker='o', color=colors)
    for i in range(len(word_inds)):
        if index2word[word_inds[i]] in filter_words:
            #print(index2word[word_inds[i]])
            plt.annotate(
                index2word[word_inds[i]].decode('utf-8', 'ignore'),
                xy=(xs[i], ys[i]), xytext=(3, 3),
                textcoords='offset points', ha='left', va='top',**TEXT_KW)


    plt.axis('off')
    plt.savefig(save_dir + '/w2v_visualization.png', bbox_inches='tight', pad_inches=0, dpi=500)

if __name__ == "__main__":
    vocab = pickle.load(open(os.path.join(args.vocab_path, 'coco_vocab.pkl'), 'rb'))
    wemb = nn.Embedding(11755, 300)
    #filter_words = ['look', 'stares', 'eats', 'stare', 'look', 'flight', 'sky', 'street', 'sitting', 'look', 'walk','car', 'road', 'blue', 'cloudy']
    filter_words = ['smoggy', 'pasted', 'away', 'drawn', 'shields', 'handful', 'visited', 'kitchen', 'climate', 'tone', 'sissors', 'tons', 'tony', '4-way', 'attacked', 'cylinder', 'tissue', 'cone', 'warthog', 'crockery', 'hang', 'hand', 'traffice', 'min', 'musical', 'trainer', 'heart-shaped', 'yamaha', 'amoco', 'lcd', 'hairless', 'cooler', 'sparse', 'night', 'cooled', 'born', 'confusing',]

    pre_init = torch.load(open('runs/gru_cross_nlp/model_best.pth.tar'))
    print("Initialize word embdding weight")
    del wemb.weight
    wemb.weight = nn.Parameter(pre_init['model'][1]['embed.weight'])
    #########################################################################################
    word_inds = [i+2 for i in range(2000)]
    visualize_embeddings(10, './', 20, wemb, vocab.idx2word, filter_words, word_inds=np.asarray(word_inds))
