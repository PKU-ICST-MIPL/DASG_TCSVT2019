# Introduction
This is the source code of our TCSVT 2019 paper "Unsupervised Cross-media Retrieval Using Domain Adaptation with Scene Graph". Please cite the following paper if you use our code.

Yuxin Peng and Jingze Chi, "Unsupervised Cross-media Retrieval Using Domain Adaptation with Scene Graph", IEEE Transactions on Circuits and Systems for Video Technology (TCSVT), DOI:10.1109/TCSVT.2019.2953692, Nov. 2019.
# Dependency

This code is implemented with pytorch.

# Data Preparation

1) The object and relation features of image is extracted by MSDN (https://github.com/yikang-li/MSDN)

2) The glove features are extracted by SGparser (https://github.com/vacancy/SceneGraphParser) and GLOVE (https://github.com/stanfordnlp/GloVe). We first obtain the object relation by SGparser, and then extracted the glove features

# Usage

There are three parts of the codes:

1. run_s：To learn the object and relation representation of the image and text

Start training and tesing by executiving the following commands:

cd ./run_s
python train.py

2. whole: To learn the representation of the whole image and text

Start training and tesing by executiving the following commands:

cd ./whole
python train.py

3. result: Merge the similarity scores for final retrieval


