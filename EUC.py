import tensorflow as tf
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score
import pickle
import os
import random

def build_mask_matrix(seqlen, valid_len_list):
    res_list = []
    base_mask = tf.ones((seqlen, seqlen)) - tf.eye(seqlen)
    # base_mask= tf.subtract(1.0, base_mask)
    bsz = len(valid_len_list)
    for i in range(bsz):
        one_base_mask = tf.identity(base_mask)
        # one_valid_len = valid_len_list[i]
        # mask = tf.cast(tf.range(seqlen) >= one_valid_len, dtype=tf.float32)
        mask = one_base_mask
        one_base_mask = tf.multiply(one_base_mask, mask)
        one_base_mask = tf.multiply(one_base_mask, tf.transpose(mask))
        res_list.append(one_base_mask)
    res_mask = tf.stack(res_list, axis=0)
    assert res_mask.shape == (bsz, seqlen, seqlen)
    return res_mask

def contrastive_loss(score_matrix, margin, bsz):
    seqlen = score_matrix.get_shape().as_list()[1]
    gold_score = tf.linalg.diag_part(score_matrix)  # bsz x seqlen
    gold_score = tf.expand_dims(gold_score, -1)
    # assert gold_score.shape == tf.TensorShape([bsz, seqlen, 1])
    difference_matrix = gold_score - score_matrix

    # assert difference_matrix.shape == tf.TensorShape([bsz, seqlen, seqlen])
    loss_matrix = margin - difference_matrix
    loss_matrix = tf.nn.relu(loss_matrix)

    base_mask = tf.ones((seqlen, seqlen)) - tf.eye(seqlen)
    loss_mask = tf.tile(tf.expand_dims(base_mask, 0), [bsz, 1, 1])
    assert loss_mask.shape == (bsz, seqlen, seqlen)

    masked_loss_matrix = loss_matrix * loss_mask
    loss_matrix = tf.reduce_sum(masked_loss_matrix, axis=-1)
    # assert loss_matrix.shape == input_ids.get_shape()
    # loss_matrix = loss_matrix * input_mask
    cl_loss = tf.reduce_sum(loss_matrix) / tf.reduce_sum(loss_mask)
    return cl_loss

def EUC(last_hidden_states, batch_size):
    # compute cl loss
    norm_rep = last_hidden_states / tf.norm(last_hidden_states, axis=-1, keepdims=True)
    cosine_scores = tf.matmul(norm_rep, norm_rep, transpose_b=True)
    # assert cosine_scores.shape == tf.TensorShape([10240, 5, 5])
    cl_loss = contrastive_loss(score_matrix=cosine_scores, margin=0.5, bsz=batch_size)
    return cl_loss
