import tensorflow as tf
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score
import pickle
import os
import random


def create_batch_norm_layer(trainable):
    return tf.layers.BatchNormalization(epsilon=1e-12, trainable=trainable)
bn_true1 = create_batch_norm_layer(trainable=True)
bn_true2 = create_batch_norm_layer(trainable=True)
bn_false1 = create_batch_norm_layer(trainable=False)
bn_false2 = create_batch_norm_layer(trainable=False)

def create_dense_layer(inner_size):
    return tf.layers.Dense(inner_size, activation=None, use_bias=False)
dense_large = create_dense_layer(inner_size = 2688*4)
dense_small = create_dense_layer(inner_size = 2688)

@tf.function
def projection_head_map(state, head_mode):
    head_mode = tf.cast(head_mode, dtype=tf.bool)
    # affine = 0
    if head_mode:
        state = dense_large(state)
        state = bn_true1(state)
        state = tf.nn.relu(state)
        state = dense_small(state)
        state = bn_true2(state)
        state = tf.nn.relu(state)
        affine = True
    else:
        state = dense_large(state)
        state = bn_false1(state)
        state = tf.nn.relu(state)
        state = dense_small(state)
        state = bn_false2(state)
        state = tf.nn.relu(state)
        affine = False
    return state, affine

def projection_head_map2(state, inner_size):
    state = tf.layers.Dense(inner_size * 4, activation=None, use_bias=False)(state)
    state = tf.layers.BatchNormalization(epsilon=1e-12, trainable=False)(state)
    state = tf.nn.relu(state)
    state = tf.layers.Dense(inner_size, activation=None, use_bias=False)(state)
    state = tf.layers.BatchNormalization(epsilon=1e-12, trainable=False)(state)
    state = tf.nn.relu(state)
    return state
