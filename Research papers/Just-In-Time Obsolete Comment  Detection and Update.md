---
authors:
  - Shanping Li
  - Zhongxin Liu
  - Xin Xia
  - David Lo
  - Meng Yan
tags:
  - LLM
  - SWE
  - CodeLLM
  - code
  - obsolescence
---
## General Idea
When writing code, our requirements may change, thus the code comments become obsolete. This is misleading for other devs because they'll rely on old information to implement new features. In this work, authors proposes a **two-stage** framework #CUP² (Comment UPdater) to solve this problem.

The proposed framework consists of two components, each of which uses a different neural network. The first one is called #OCD (**Obsolete Comment Detector**), while the second is called #CUP (**Comment UPdater**).

By proposing a special neural #encoder, authors were able to **combine** a code change and an associated comment as feature vectors. OCD leverages an **attention-based** output layer to predict the probability that a comment should be updated.
CUP on its side, uses the same neural encoder combined with an **RNN-based** decoder to generate updated comments.

CUP learns simultaneously representations for code changes and their comments.  ****

## Approach
The whole CUP² idea can be summarized as follows:
Given $t, t', x$ and $y$ respectively named *old code, new code, old comment* and *new comment*, the goal is to learn two methods: *detect* and *update*.
*detect* is defined as follows:

$$
detect(t, t', x) =\begin{cases}
1, \text{if } x \ne y \\
0, \text{otherwise} 
\end{cases}
$$
The *update* method is define as follows:
$$
update(t, t',x) = y
$$

