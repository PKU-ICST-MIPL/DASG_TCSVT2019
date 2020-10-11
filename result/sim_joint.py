import numpy as np

def i2t(imagesLen, sims, npts=None, return_ranks=False):
    """
    Images->Text (Image Annotation)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    outputR = open('imgRank.txt', 'w')
    outputRes = open('imgRes.txt', 'w')

    npts = imagesLen
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]

        scoreTmp = ','.join([str(x) for x in inds[0:5]])
        outputR.write(scoreTmp+'\n')
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < 5:
                outputRes.write(str(index)+',')

            if tmp < rank:
                rank = tmp
                
        ranks[index] = rank
        top1[index] = inds[0]

        outputRes.write('\n')

    outputR.close()
    outputRes.close()
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


def t2i(imagesLen, sims, npts=None, return_ranks=False):
    """
    Text->Images (Image Search)
    Images: (N, n_region, d) matrix of images
    Captions: (5N, max_n_word, d) matrix of captions
    CapLens: (5N) array of caption lengths
    sims: (N, 5N) matrix of similarity im-cap
    """
    outputR = open('txtRank.txt', 'w')
    outputRes = open('txtRes.txt', 'w')

    npts = imagesLen
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)

    # --> (5N(caption), N(image))
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]

            scoreTmp = ','.join([str(x) for x in inds[0:10]])
            outputR.write(scoreTmp+'\n')
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

            if(ranks[5 * index + i]==0):
                outputRes.write(str(5 * index + i)+'\n')

    outputR.close()
    outputRes.close()
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


def validate(sims):

    imageLen = sims.shape[0]
    # caption retrieval
    (r1, r5, r10, medr, meanr) = i2t(imageLen, sims)

    # image retrieval
    (r1i, r5i, r10i, medri, meanr) = t2i(imageLen, sims)


    currscore = r1 + r5 + r10 + r1i + r5i + r10i
    print("Image to text: %.1f, %.1f, %.1f" % (r1, r5, r10))
    print("Text to image: %.1f, %.1f, %.1f" % (r1i, r5i, r10i))

    return currscore

if __name__ == '__main__':

    sims1 = np.load('/home/chijingze/UDAG/result/flickr_sims.npy')
    sims2 = np.load('/home/chijingze/UDAG/result/d_i2t_flickr.npy')
    # print(sims2.shape)
    # print(validate(sims1))
    # print(validate(sims2))

    # for i in range(10):

    i = 6
    a = i*0.1
    b = (10-i)*0.1
    sims = a*sims1+b*sims2
    print(validate(sims))