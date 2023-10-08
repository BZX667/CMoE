import tensorflow as tf
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score
import pickle
import os
import random

def mask_correlated_samples(batch_size):
    N = 2 * batch_size
    # 构造全1矩阵
    mask = tf.ones((N, N), dtype=tf.bool, name='mask')
    # 构造对角线为0的掩码矩阵
    diag = tf.ones((N,), dtype=tf.bool)
    mask = tf.linalg.set_diag(mask, diag)
    # 创建索引和值矩阵
    indices = tf.concat([tf.range(batch_size), tf.range(batch_size)], axis=0)
    updates = tf.concat([tf.zeros(batch_size, dtype=bool), tf.ones(batch_size, dtype=bool)], axis=0)
    # 根据索引和值矩阵创建更新矩阵
    update_matrix = tf.scatter_nd(tf.expand_dims(indices, axis=1), tf.expand_dims(updates, axis=1), shape=[N, 1])
    # 对全1矩阵和更新矩阵进行逻辑与运算
    mask = tf.logical_and(mask, update_matrix)
    return mask

def cts_loss_diff_samples(z_i, z_j, temp, batch_size):  # B * D    B * D
    # z_i = tf.nn.l2_normalize(z_i, axis=1)
    # z_j = tf.nn.l2_normalize(z_j, axis=1)

    N = 2 * batch_size
    z = tf.concat([z_i, z_j], axis=0)  # 2B * D

    sim = tf.matmul(z, tf.transpose(z)) / temp  # 2B * 2B
    sim_i_j = tf.matrix_diag_part(sim[:batch_size, batch_size:2*batch_size])  # B * 1
    sim_j_i = tf.matrix_diag_part(sim[batch_size:2*batch_size, :batch_size])  # B * 1

    positive_samples = tf.concat([sim_i_j, sim_j_i], axis=0)
    positive_samples = tf.reshape(positive_samples, [N, 1])

    mask = mask_correlated_samples(batch_size)

    negative_samples = tf.boolean_mask(sim, mask)
    negative_samples = tf.reshape(negative_samples, [N, -1])

    labels = tf.zeros(N, dtype=tf.int32)
    logits = tf.concat([positive_samples, negative_samples], axis=1)  # N * 2

    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(loss)
    return loss
