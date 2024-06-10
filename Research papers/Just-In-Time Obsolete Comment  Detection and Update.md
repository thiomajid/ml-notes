---
author: " Zhongxin Liu, Xin Xia, David Lo, Meng Yan, Shanping Li"
---
## General Idea
When writing code, our requirements may change, thus the code comments become obsolete. This is misleading for other devs because they'll rely on old information to implement new features. In this work, authors proposes a **two-stage** framework #CUP (Comment UPdater) to solve this problem.

The proposed framework consists of two components, each of which uses a different neural network. The first one is called #OCD (Obsolete Comment Detector), while the second is called #CUP (Comment UPdater).

By proposing a special neural #encoder, authors were able to **combine** a code change and an associated comment as feature vectors. Using probably a #BERT style architecture, OCD leverages an **attention-based** output layer to predict the probability that a comment should be updated.

CUP on its side, uses the same neural encoder combined with an **RNN-based** decoder to generate updated comments.
