---
title: "Deep Convolutional Neural Networks: DenseNet"
excerpt: "Introduction to DenseNet"
author: Praneeth Bellamkonda
date: 2019-03-01
tags: [Deep Learning, CNN, Computer Vision]
header:
    image: "/images/densenet/deeplearning.jpg"
---

{% include base_path %}
{% include toc %}

Convolutional Neural Networks are the popular choice of neural networks for different Computer Vision tasks such as image recognition.
The problems arise with CNNs when they go deeper. This is because the path for information from the input layer until the output layer (and for the gradient in the opposite direction) becomes so big, that they can get vanished before reaching the other side.

To solve this problem [Gao Huang et al](https://arxiv.org/abs/1608.06993). introduced Dense Convolutional networks. DenseNets have several compelling advantages:

*   alleviate the vanishing-gradient problem
*   strengthen feature propagation
*   encourage feature reuse, and substantially reduce the number of parameters.
[Source: https://theailearner.com](https://theailearner.com/2018/12/09/densely-connected-convolutional-networks-densenet/)








