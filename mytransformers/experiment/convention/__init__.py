"""Convolutions with Attention.

In this module, A self attention based convolutional block is implemented. The main
idea is to use convolutional layers instead of linear layers in the attention head of
the paper "Attention is all you need." This perspective leads to other ways of
implementing the common operations on images. For example the ClassificationHead,
SelfAttentionFixedResize and SelfAttentionResize are implemented using this mindset.
"""
from . import body, head, positionalencoder, resize
