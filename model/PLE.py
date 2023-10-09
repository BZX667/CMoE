#!/usr/bin/env python
# coding=gbk

# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import shutil
import os
import json
import glob
from datetime import date, timedelta
from time import time
import datetime
import numpy as np
import random
import tensorflow as tf
import logging
from tensorflow.keras.layers import Dense, Lambda
from core import DNN, PredictionLayer, reduce_sum
from sklearn.feature_selection import mutual_info_regression

import sys
sys.path.append("..")
import data_augment
import EAC
import EUC
import project_head

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
slim = tf.contrib.slim

#################### CMD Arguments ######################
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("dist_mode", 0, "distribuion mode {0-loacal, 1-single_dist, 2-multi_dist}")
tf.app.flags.DEFINE_string("ps_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("worker_hosts", '', "Comma-separated list of hostname:port pairs")
tf.app.flags.DEFINE_string("job_name", '', "One of 'ps', 'worker'")
tf.app.flags.DEFINE_integer("task_index", 0, "Index of task within the job")
tf.app.flags.DEFINE_integer("num_threads", 16, "Number of threads")
tf.app.flags.DEFINE_integer("field_size", 0, "Number of common fields")
tf.app.flags.DEFINE_integer("embedding_size", 32, "Embedding size")
tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of epochs")
tf.app.flags.DEFINE_integer("batch_size", 64, "Number of batch size")
tf.app.flags.DEFINE_integer("log_steps", 1000, "save summary every steps")
tf.app.flags.DEFINE_float("learning_rate", 0.0005, "learning rate")
tf.app.flags.DEFINE_float("l2_reg", 0.0000, "L2 regularization")
tf.app.flags.DEFINE_string("loss_type", 'log_loss', "loss type {square_loss, log_loss}")
tf.app.flags.DEFINE_float("ctr_task_wgt", 0.5, "loss weight of ctr task")
tf.app.flags.DEFINE_string("optimizer", 'Adam', "optimizer type {Adam, Adagrad, GD, Momentum}")
tf.app.flags.DEFINE_string("deep_layers", '256,128,64', "deep layers")
tf.app.flags.DEFINE_string("dropout", '0.5,0.5,0.5', "dropout rate")
tf.app.flags.DEFINE_boolean("batch_norm", False, "perform batch normaization (True or False)")
tf.app.flags.DEFINE_float("batch_norm_decay", 0.9, "decay for the moving average(recommend trying decay=0.9)")
tf.app.flags.DEFINE_string("data_dir", '/home/dc/deeplearning/DeepFM/DIN/DIN_data/', "data dir")
tf.app.flags.DEFINE_string("dt_dir", '', "data dt partition")
tf.app.flags.DEFINE_string("model_dir", '/home/dc/mao/model_ckpt/', "model check point dir")
tf.app.flags.DEFINE_string("servable_model_dir", '/home/dc/mao/servable_model/',
                           "export servable model for TensorFlow Serving")
tf.app.flags.DEFINE_string("task_type", 'train', "task type {train, infer, eval, export}")
tf.app.flags.DEFINE_boolean("clear_existing_model", False, "clear existing model or not")
tf.app.flags.DEFINE_integer("feature_size", 500000, "feature_size_threshold")
tf.app.flags.DEFINE_boolean("cl_loss", True, "True or False")

# paras
# cl_loss = True
expert_dnn_hidden_units=(256,128)
gate_dnn_hidden_units=()
tower_dnn_hidden_units=(32,)
seed = 1024
tf.set_random_seed(666)
random.seed(123)
np.random.seed(123)

specific_expert_num = 4
shared_expert_num = 8
num_levels = 1

l2_reg_dnn = 0
dnn_dropout = 0.8
l2_reg_embedding = 0.00001
dnn_activation = 'relu'
dnn_use_bn = False
task_types = ('binary', 'binary', 'binary')
task_names = ('ctr', 'ctcvr', 'cdr')
num_tasks = len(task_names)

def input_fn(filenames, batch_size=32, num_epochs=1, perform_shuffle=True):
    print('Parsing', filenames)

    def _parse_fn(record):
        features = {
            "y": tf.FixedLenFeature([], tf.float32),
            "feat_ids": tf.FixedLenFeature([FLAGS.field_size], tf.int64),
            "feat_vals": tf.FixedLenFeature([FLAGS.field_size], tf.float32),
            "u_ticketids": tf.VarLenFeature(tf.int64),
            "u_ticketvals": tf.VarLenFeature(tf.float32),
            "u_tag1ids": tf.VarLenFeature(tf.int64),
            "u_tag2ids": tf.VarLenFeature(tf.int64),
            "labels_id": tf.VarLenFeature(tf.int64),
            "z": tf.FixedLenFeature([], tf.float32),
            "d": tf.FixedLenFeature([], tf.float32),
            "a_ticket_ids": tf.FixedLenFeature([], tf.int64),
            "a_tag1_ids": tf.FixedLenFeature([], tf.int64),
            "a_tag2_ids": tf.FixedLenFeature([], tf.int64),
            "u_ticketids_cvr": tf.VarLenFeature(tf.int64),
            "u_ticketvals_cvr": tf.VarLenFeature(tf.float32),
            "u_tag1ids_cvr": tf.VarLenFeature(tf.int64),
            "u_tag2ids_cvr": tf.VarLenFeature(tf.int64),
            "labels_id_cvr": tf.VarLenFeature(tf.int64),
        }
        parsed = tf.parse_single_example(record, features)
        y = parsed.pop('y')
        z = parsed.pop('z')
        d = parsed.pop('d')
        return parsed, {"y": y, "z": z, "d": d}

    # Extract lines from input files using the Dataset API, can pass one filename or filename list
    dataset = tf.data.TFRecordDataset(filenames).map(_parse_fn, num_parallel_calls=10).prefetch(
        500000)  # multi-thread pre-process then prefetch

    # Randomizes input using a window of 256 elements (read into memory)
    if perform_shuffle:
        dataset = dataset.shuffle(buffer_size=256)

    # epochs from blending together.
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)  # Batch size to use
    iterator = dataset.make_one_shot_iterator()
    batch_features, batch_labels = iterator.get_next()
    return batch_features, batch_labels


def model_fn(features, labels, mode, params):
    # ------hyperparameters----
    field_size = params["field_size"]
    feature_size_thre = params["feature_size"]
    embedding_size = params["embedding_size"]
    l2_reg = params["l2_reg"]
    learning_rate = params["learning_rate"]
    common_dims = field_size * embedding_size

    # ------build feaure-------
    feat_ids = features['feat_ids']
    feat_ids = tf.reshape(feat_ids, shape=[-1, field_size])
    feat_vals = features['feat_vals']
    feat_vals = tf.reshape(feat_vals, shape=[-1, field_size])

    # {User multi-hot}
    u_ticketids = features['u_ticketids']
    u_ticketvals = features['u_ticketvals']
    u_tag1ids = features['u_tag1ids']
    u_tag1vals = features['u_ticketvals']
    u_tag2ids = features['u_tag2ids']
    u_tag2vals = features['u_ticketvals']
    labels_id = features['labels_id']
    labelsvals = features['u_ticketvals']

    u_ticketids_cvr = features['u_ticketids_cvr']
    u_ticketvals_cvr = features['u_ticketvals_cvr']
    u_tag1ids_cvr = features['u_tag1ids_cvr']
    u_tag1vals_cvr = features['u_ticketvals_cvr']
    u_tag2ids_cvr = features['u_tag2ids_cvr']
    u_tag2vals_cvr = features['u_ticketvals_cvr']
    labels_id_cvr = features['labels_id_cvr']
    labelsvals_cvr = features['u_ticketvals_cvr']

    if (mode == tf.estimator.ModeKeys.TRAIN) or (mode == tf.estimator.ModeKeys.EVAL):
        y = labels['y']
        z = labels['z']
        d = labels['d']

    # ------build f(x)------
    with tf.variable_scope("Shared-Embedding-layer"):
        Feat_Emb = tf.get_variable(name='embeddings', shape=[feature_size_thre, embedding_size],
                                   initializer=tf.glorot_normal_initializer())
        common_embs = tf.nn.embedding_lookup(Feat_Emb, feat_ids)  # None * F * K
        # reshape，保证其第一第二维度和embedding一致
        feat_vals = tf.reshape(feat_vals, shape=[-1, field_size, 1])
        common_embs = tf.multiply(common_embs, feat_vals)  # None * F * K
        sum_square = tf.square(tf.reduce_sum(common_embs, 1))  # None * K
        square_sum = tf.reduce_sum(tf.square(common_embs), 1)  # None * K
        y_train_new = tf.subtract(sum_square, square_sum)


        u_ticket_emb = tf.nn.embedding_lookup_sparse(Feat_Emb, sp_ids=u_ticketids, sp_weights=u_ticketvals,combiner="sum")
        u_tag1_emb = tf.nn.embedding_lookup_sparse(Feat_Emb, sp_ids=u_tag1ids, sp_weights=u_tag1vals, combiner="sum")
        u_tag2_emb = tf.nn.embedding_lookup_sparse(Feat_Emb, sp_ids=u_tag2ids, sp_weights=u_tag2vals, combiner="sum")
        u_label_emb = tf.nn.embedding_lookup_sparse(Feat_Emb, sp_ids=labels_id, sp_weights=labelsvals, combiner="sum")
        # cvr embedding
        u_ticket_cvr_emb = tf.nn.embedding_lookup_sparse(Feat_Emb, sp_ids=u_ticketids_cvr, sp_weights=u_ticketvals_cvr,
                                                         combiner="sum")
        u_tag1_cvr_emb = tf.nn.embedding_lookup_sparse(Feat_Emb, sp_ids=u_tag1ids_cvr, sp_weights=u_tag1vals_cvr,
                                                       combiner="sum")
        u_tag2_cvr_emb = tf.nn.embedding_lookup_sparse(Feat_Emb, sp_ids=u_tag2ids_cvr, sp_weights=u_tag2vals_cvr,
                                                       combiner="sum")
        u_label_cvr_emb = tf.nn.embedding_lookup_sparse(Feat_Emb, sp_ids=labels_id_cvr, sp_weights=labelsvals_cvr,
                                                        combiner="sum")

        x_concat = tf.concat(
            [tf.reshape(common_embs, shape=[-1, common_dims]), u_ticket_emb, u_tag1_emb, u_tag2_emb, u_label_emb,
             u_ticket_cvr_emb, u_tag1_cvr_emb, u_tag2_cvr_emb, u_label_cvr_emb, y_train_new], axis=1)

    def cgc_net(inputs, level_name, is_last=False):
        specific_expert_outputs = []
        for i in range(num_tasks):
            for j in range(specific_expert_num):
                expert_network = DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                                     seed=seed,
                                     name=level_name + 'task_' + task_names[i] + '_expert_specific_' + str(j))(inputs[i])
                specific_expert_outputs.append(expert_network)

        # build task-shared expert layer
        shared_expert_outputs = []
        for k in range(shared_expert_num):
            expert_network = DNN(expert_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                                 seed=seed,
                                 name=level_name + 'expert_shared_' + str(k))(inputs[-1])
            shared_expert_outputs.append(expert_network)

        cgc_outs = []
        for i in range(num_tasks):
            # concat task-specific expert and task-shared expert
            cur_expert_num = specific_expert_num + shared_expert_num
            # task_specific + task_shared
            cur_experts = specific_expert_outputs[i * specific_expert_num:(i + 1) * specific_expert_num] + shared_expert_outputs
            expert_concat = Lambda(lambda x: tf.stack(x, axis=1))(cur_experts)
            # expert_concat2 = Lambda(lambda x: tf.stack(x, axis=1))(shared_expert_outputs)
            # build gate layers
            gate_input = DNN(gate_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                             seed=seed,
                             name=level_name + 'gate_specific_' + task_names[i])(inputs[i])  # gate[i] for task input[i]
            gate_out = Dense(cur_expert_num, use_bias=False, activation='softmax',
                                             name=level_name + 'gate_softmax_specific_' + task_names[i])(gate_input)
            gate_out = Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)

            # gate multiply the expert
            gate_mul_expert = Lambda(lambda x: reduce_sum(x[0] * x[1], axis=1, keep_dims=False),
                                                     name=level_name + 'gate_mul_expert_specific_' + task_names[i])([expert_concat, gate_out])
            cgc_outs.append(gate_mul_expert)

        # task_shared gate, if the level not in last, add one shared gate
        if not is_last:
            cur_expert_num = num_tasks * specific_expert_num + shared_expert_num
            cur_experts = specific_expert_outputs + shared_expert_outputs  # all the expert include task-specific expert and task-shared expert

            expert_concat = Lambda(lambda x: tf.stack(x, axis=1))(cur_experts)
            # expert_concat2 = Lambda(lambda x: tf.stack(x, axis=1))(shared_expert_outputs)

            # build gate layers
            gate_input = DNN(gate_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn,
                             seed=seed,
                             name=level_name + 'gate_shared')(inputs[-1])  # gate for shared task input

            gate_out = Dense(cur_expert_num, use_bias=False, activation='softmax',
                                             name=level_name + 'gate_softmax_shared')(gate_input)
            gate_out = Lambda(lambda x: tf.expand_dims(x, axis=-1))(gate_out)

            # gate multiply the expert
            gate_mul_expert = Lambda(lambda x: reduce_sum(x[0] * x[1], axis=1, keep_dims=False),
                                                     name=level_name + 'gate_mul_expert_shared')(
                [expert_concat, gate_out])

            cgc_outs.append(gate_mul_expert)
        return cgc_outs, expert_concat

    with tf.variable_scope("ple_out"):
        # build Progressive Layered Extraction
        ple_inputs = [x_concat] * (num_tasks + 1)  # [task1, task2, ... taskn, shared task]
        ple_outputs = []
        for i in range(num_levels):
            if i == num_levels - 1:  # the last level
                ple_outputs, expert_concat = cgc_net(inputs=ple_inputs, level_name='level_' + str(i) + '_', is_last=True)
            else:
                ple_outputs, expert_concat = cgc_net(inputs=ple_inputs, level_name='level_' + str(i) + '_', is_last=False)
                ple_inputs = ple_outputs

    with tf.variable_scope("MTL-Layer"):
        task_outs = []
        for task_type, task_name, ple_out in zip(task_types, task_names, ple_outputs):
            # build tower layer
            tower_output = DNN(tower_dnn_hidden_units, dnn_activation, l2_reg_dnn, dnn_dropout, dnn_use_bn, seed=seed,
                               name='tower_' + task_name)(ple_out)
            logit = Dense(1, use_bias=False)(tower_output)
            output = PredictionLayer(task_type, name=task_name)(logit)
            task_outs.append(output)

        y_ctr = task_outs[0]
        y_cvr = task_outs[1]
        y_cdr = task_outs[2]
        pctr = y_ctr
        pcvr = y_cvr
        pcdr = y_cdr
        pctcvr = pctr * pcvr
        pcdcvr = pcvr * pcdr
        pctcvcdr = pctr * pcvr * pcdr
        output = tf.concat([pctr, pcvr, pctcvr], axis=-1, name='output_preb')

    predictions = {"pcvr": pcvr, "pctr": pctr, "pctcvr": pctcvr}
    export_outputs = {
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: tf.estimator.export.PredictOutput(
            predictions)}

    # Provide an estimator spec for `ModeKeys.PREDICT`
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs=export_outputs)

    with tf.variable_scope("loss_weight"):
        # ------bulid loss------
        ctr_loss = tf.reduce_mean(tf.losses.log_loss(predictions=pctr, labels=y))
        cvr_loss = tf.reduce_mean(tf.losses.log_loss(predictions=pctcvr, labels=z))
        cdr_loss = tf.reduce_mean(tf.losses.log_loss(predictions=pctcvcdr, labels=d))
        loss = ctr_loss + cvr_loss + cdr_loss + l2_reg * tf.nn.l2_loss(Feat_Emb)

    eval_metric_ops = {
        # "CTCVR_AUC": tf.metrics.auc(z, pctcvr),
        "CTCVCDR_AUC": tf.metrics.auc(d, pctcvcdr),
        "CDR_AUC": tf.metrics.auc(d, pcdr),
        "CVR_AUC": tf.metrics.auc(z, pcvr),
        "CTR_AUC": tf.metrics.auc(y, pctr),
        # "CDCVR_AUC": tf.metrics.auc(d, pcdcvr)
    }

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            eval_metric_ops=eval_metric_ops)

    # ------bulid optimizer------
    if FLAGS.optimizer == 'Adam':
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
    elif FLAGS.optimizer == 'Adagrad':
        optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate, initial_accumulator_value=1e-8)
    elif FLAGS.optimizer == 'Momentum':
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.95)
    elif FLAGS.optimizer == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(learning_rate)

    main_task_train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    if mode == tf.estimator.ModeKeys.TRAIN:
        if FLAGS.cl_loss:
            expert_loss=True
            sample_loss=True
            optimizer_num = 3 if expert_loss and sample_loss else 2 if expert_loss or sample_loss else 1
            inner_size = x_concat.get_shape().as_list()[1]
            step = tf.divide(tf.train.get_global_step(), optimizer_num)
            head_mode = tf.cond(tf.equal(tf.mod(step, 2), 0), lambda: 0, lambda: 1)
            # head_mode_print = tf.Print(head_mode, [head_mode], "head_mode: %.16f")
            # head_mode_print2 = tf.Print(1-head_mode, [1-head_mode], "head_mode: %.16f")
            step_print = tf.Print(step, [step], "step: %.16f")
            x_concat_map, affine1 = project_head.projection_head_map(x_concat, head_mode)

            x_concat_noise = data_augment.add_gaussian_noise(x_concat, std=0.01)

            x_concat_drop = data_augment.random_dropout_embedding(x_concat_noise, keep_prob=0.95)
            # x_concat_cfm = data_augment.cfm_masking_F_F(x_concat, 0.1, inner_size)
            # x_concat_cfm, mask_ind = data_augment.cfm_masking_F_F_feat(x_concat_cfm, feat_ids, 0.5)
            x_concat_cfm_map, affine2 = project_head.projection_head_map(x_concat_drop, 1-head_mode)
            affine1_print = tf.Print(affine1, [affine1], "affine1: %s" % affine1)
            affine2_print = tf.Print(affine2, [affine2], "affine2: %s" % affine2)

            # x_concat_cfm = data_augment.cfm_masking_F_L(x_concat_cfm, 0.05, y)
            # x_concat_noise = data_augment.add_uniform_noise(x_concat, weight=0.001)
            # x_concat_noise_map = project_head.projection_head_map(x_concat_noise, inner_size, mode)
            # input_ids = tf.ones(shape=(FLAGS.batch_size, FLAGS.num_experts), dtype=tf.float32)
            # learning_rate3 = tf.train.piecewise_constant_decay(tf.train.get_global_step()
            #                                                    ,boundaries=[100, 200, 300, 400, 500]
            #                                                    ,values=[0.0006, 0.0006/2, 0.0006/4, 0.0006/8, 0.0006/16, 0.0006/32])
            # learning_rate_exp = tf.train.exponential_decay(0.006, global_step=tf.train.get_global_step(), decay_steps=3, decay_rate=0.96,staircase=True)
            # lr_print = tf.Print(learning_rate_exp, [learning_rate_exp], "lr: %.16f")

            if (expert_loss & sample_loss):
                # expert_concat_map = loss_function.projection_head_map2(expert_concat, expert_dnn_hidden_units[-1])
                # expert_concat = Lambda(lambda x: tf.stack(x, axis=1))(shared_expert_outputs)
                EUC_loss = 0.3 * EUC.EUC(last_hidden_states=expert_concat, batch_size=FLAGS.batch_size)
                EUC_print = tf.Print(EUC_loss, [EUC_loss], "EUC_loss: %.16f")
                EUC_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate*1/3, beta1=0.9, beta2=0.999, epsilon=1e-8)
                EUC_train_op = EUC_optimizer.minimize(EUC_loss, global_step=tf.train.get_global_step())

                EAC_loss = 0.4 * EAC.EAC(z_i=x_concat_map, z_j=x_concat_cfm_map, temp=0.01, batch_size=FLAGS.batch_size)
                EAC_print = tf.Print(EAC_loss, [EAC_loss], "EAC_loss: %.16f")
                EAC_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate*1/3, beta1=0.9, beta2=0.999, epsilon=1e-8)
                EAC_train_op = EAC_optimizer.minimize(EAC_loss, global_step=tf.train.get_global_step())
                main_task_train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
                train_op = tf.group(main_task_train_op, EUC_train_op, EAC_train_op)
                hooks = [
                    tf.train.LoggingTensorHook(
                        {
                            "EUC_loss": EUC_print,
                            "EAC_loss": EAC_print,
                            "step": step_print,
                            "sample1": affine1_print,
                            "sample2": affine2_print,
                            # "lr": lr_print
                        },
                        every_n_iter=1
                    )
                ]

            elif expert_loss:
                # expert_concat = Lambda(lambda x: tf.stack(x, axis=1))(shared_expert_outputs)
                EUC_loss = EUC.EUC(last_hidden_states=expert_concat, batch_size=FLAGS.batch_size)
                EUC_print = tf.Print(EUC_loss, [EUC_loss], "EUC_loss: %.16f")
                EUC_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate*1/3, beta1=0.9, beta2=0.999, epsilon=1e-8)
                EUC_train_op = EUC_optimizer.minimize(EUC_loss, global_step=tf.train.get_global_step())
                train_op = tf.group(main_task_train_op, EUC_train_op)
                hooks = [tf.train.LoggingTensorHook({"EUC_loss": EUC_print}, every_n_iter=10)]

            elif sample_loss:
                EAC_loss = EAC.EAC(z_i=x_concat_map, z_j=x_concat_cfm_map,temp=0.01,batch_size=FLAGS.batch_size)
                EAC_print = tf.Print(EAC_loss, [EAC_loss], "EAC_loss: %.16f")
                EAC_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate*1/3, beta1=0.9,beta2=0.999, epsilon=1e-8)
                EAC_train_op = EAC_optimizer.minimize(EAC_loss,global_step=tf.train.get_global_step())
                train_op = tf.group(main_task_train_op, EAC_train_op)
                hooks = [
                    tf.train.LoggingTensorHook(
                        {
                            "EAC_loss": EAC_print,
                            "step": step_print,
                            "sample1": affine1_print,
                            "sample2": affine2_print,
                            # "lr": lr_print
                        },
                        every_n_iter=1
                    )
                ]

        else:
            train_op = main_task_train_op
            hooks = None

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=loss,
            train_op=train_op,
            scaffold=None,
            training_hooks=hooks)


def batch_norm_layer(x, train_phase, scope_bn):
    bn_train = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=True, reuse=None, scope=scope_bn)
    bn_infer = tf.contrib.layers.batch_norm(x, decay=FLAGS.batch_norm_decay, center=True, scale=True,
                                            updates_collections=None, is_training=False, reuse=True, scope=scope_bn)
    z = tf.cond(tf.cast(train_phase, tf.bool), lambda: bn_train, lambda: bn_infer)
    return z


def set_dist_env():
    if FLAGS.dist_mode == 1:
        ps_hosts = FLAGS.ps_hosts.split(',')
        chief_hosts = FLAGS.chief_hosts.split(',')
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        print('ps_host', ps_hosts)
        print('chief_hosts', chief_hosts)
        print('job_name', job_name)
        print('task_index', str(task_index))
        tf_config = {
            'cluster': {'chief': chief_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index}
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)
    elif FLAGS.dist_mode == 2:
        ps_hosts = FLAGS.ps_hosts.split(',')
        worker_hosts = FLAGS.worker_hosts.split(',')
        chief_hosts = worker_hosts[0:1]  # get first worker as chief
        worker_hosts = worker_hosts[2:]  # the rest as worker
        task_index = FLAGS.task_index
        job_name = FLAGS.job_name
        print('ps_host', ps_hosts)
        print('worker_host', worker_hosts)
        print('chief_hosts', chief_hosts)
        print('job_name', job_name)
        print('task_index', str(task_index))
        # use #worker=0 as chief
        if job_name == "worker" and task_index == 0:
            job_name = "chief"
        # use #worker=1 as evaluator
        if job_name == "worker" and task_index == 1:
            job_name = 'evaluator'
            task_index = 0
        # the others as worker
        if job_name == "worker" and task_index > 1:
            task_index -= 2

        tf_config = {
            'cluster': {'chief': chief_hosts, 'worker': worker_hosts, 'ps': ps_hosts},
            'task': {'type': job_name, 'index': task_index}
        }
        print(json.dumps(tf_config))
        os.environ['TF_CONFIG'] = json.dumps(tf_config)


def main(_):
    # ------check Arguments------
    if FLAGS.dt_dir == "":
        FLAGS.dt_dir = (date.today() + timedelta(-2)).strftime('%Y-%m-%d')

    # yesterday_0 = FLAGS.dt_dir
    # day_before_yest_0 = (datetime.date.today() + datetime.timedelta(days=-1)).strftime('%Y-%m-%d')

    FLAGS.model_dir = FLAGS.model_dir + 'model_ckpt_ple'

    tr_data_dirs = ['/data/ESSM/2023-07-01_tfrecord_mmoe_2']
    va_data_dirs = ['/data/double_price/2023-07-02_tfrecord_cvr_2']

    # 初始化一个空列表来存储所有的文件
    tr_files = []
    for data_dir in tr_data_dirs:
        # 使用通配符来匹配文件夹中的所有文件
        files = glob.glob("{0}/tr*.tfrecord".format(data_dir))
        tr_files.extend(files)
    random.shuffle(tr_files)
    print("tr_files:", tr_files)

    va_files = []
    for data_dir in va_data_dirs:
        # 使用通配符来匹配文件夹中的所有文件
        files = glob.glob("%s/tr*tfrecord" % data_dir)
        va_files.extend(files)
    random.shuffle(va_files)
    print("va_files:", va_files)

    print('task_type ', FLAGS.task_type)
    print('model_dir ', FLAGS.model_dir)
    print('data_dir ', FLAGS.data_dir)
    print('dt_dir ', FLAGS.dt_dir)
    print('num_epochs ', FLAGS.num_epochs)
    print('feature_size ', FLAGS.feature_size)
    print('field_size ', FLAGS.field_size)
    print('embedding_size ', FLAGS.embedding_size)
    print('batch_size ', FLAGS.batch_size)
    print('deep_layers ', FLAGS.deep_layers)
    print('dropout ', FLAGS.dropout)
    print('loss_type ', FLAGS.loss_type)
    print('optimizer ', FLAGS.optimizer)
    print('learning_rate ', FLAGS.learning_rate)
    print('l2_reg ', FLAGS.l2_reg)
    print('ctr_task_wgt ', FLAGS.ctr_task_wgt)

    if FLAGS.clear_existing_model:
        try:
            shutil.rmtree(FLAGS.model_dir)
        except Exception as e:
            print(e, "at clear_existing_model")
        else:
            print("existing model cleaned at %s" % FLAGS.model_dir)

    set_dist_env()

    # ------bulid Tasks------
    model_params = {
        "field_size": FLAGS.field_size,
        "feature_size": FLAGS.feature_size,
        "embedding_size": FLAGS.embedding_size,
        "learning_rate": FLAGS.learning_rate,
        "l2_reg": FLAGS.l2_reg,
        "deep_layers": FLAGS.deep_layers,
        "dropout": FLAGS.dropout,
        "ctr_task_wgt": FLAGS.ctr_task_wgt
    }
    config = tf.estimator.RunConfig(tf_random_seed=1234).replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0, 'CPU': FLAGS.num_threads}),
        log_step_count_steps=FLAGS.log_steps, save_summary_steps=FLAGS.log_steps)
    Estimator = tf.estimator.Estimator(model_fn=model_fn, model_dir=FLAGS.model_dir, params=model_params, config=config)

    if FLAGS.task_type == 'train':
        Estimator.train(input_fn=lambda: input_fn(tr_files, num_epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size))
    elif FLAGS.task_type == 'eval':
        Estimator.evaluate(input_fn=lambda: input_fn(va_files, num_epochs=1, batch_size=FLAGS.batch_size))
    elif FLAGS.task_type == 'infer':
        preds = Estimator.predict(input_fn=lambda: input_fn(te_files, num_epochs=1, batch_size=FLAGS.batch_size),predict_keys="prob")
        with open(FLAGS.data_dir + "/pred.txt", "w") as fo:
            for prob in preds:
                fo.write("%f\t%f\n" % (prob['pctr'], prob['pcvr']))
    elif FLAGS.task_type == 'export':
        input_feat_ids = tf.placeholder(tf.int64, shape=[None, FLAGS.field_size], name='feat_ids')
        input_feat_vals = tf.placeholder(tf.float32, shape=[None, FLAGS.field_size], name='feat_vals')
        input_u_ticketids = tf.sparse_placeholder(tf.int64, name='u_ticketids')
        input_u_ticketvals = tf.sparse_placeholder(tf.float32, name='u_ticketvals')
        input_u_tag1ids = tf.sparse_placeholder(tf.int64, name='u_tag1ids')
        input_u_tag2ids = tf.sparse_placeholder(tf.int64, name='u_tag2ids')
        input_labels_id = tf.sparse_placeholder(tf.int64, name='labels_id')

        input_u_ticketids_cvr = tf.sparse_placeholder(tf.int64, name='u_ticketids_cvr')
        input_u_ticketvals_cvr = tf.sparse_placeholder(tf.float32, name='u_ticketvals_cvr')
        input_u_tag1ids_cvr = tf.sparse_placeholder(tf.int64, name='u_tag1ids_cvr')
        input_u_tag2ids_cvr = tf.sparse_placeholder(tf.int64, name='u_tag2ids_cvr')
        input_labels_id_cvr = tf.sparse_placeholder(tf.int64, name='labels_id_cvr')

        input_a_ticket_ids = tf.placeholder(tf.int64, [], name='a_ticket_ids')
        input_a_tag1_ids = tf.placeholder(tf.int64, [], name='a_tag1_ids')
        input_a_tag2_ids = tf.placeholder(tf.int64, [], name='a_tag2_ids')

        features = {
            "feat_ids": input_feat_ids,
            "feat_vals": input_feat_vals,
            "u_ticketids": input_u_ticketids,
            "u_ticketvals": input_u_ticketvals,
            "u_tag1ids": input_u_tag1ids,
            "u_tag2ids": input_u_tag2ids,
            "labels_id": input_labels_id,
            "a_ticket_ids": input_a_ticket_ids,
            "a_tag1_ids": input_a_tag1_ids,
            "a_tag2_ids": input_a_tag2_ids,
            "u_ticketids_cvr": input_u_ticketids_cvr,
            "u_ticketvals_cvr": input_u_ticketvals_cvr,
            "u_tag1ids_cvr": input_u_tag1ids_cvr,
            "u_tag2ids_cvr": input_u_tag2ids_cvr,
            "labels_id_cvr": input_labels_id_cvr
        }
        receiver_tensors = {'feats_id': input_feat_ids}
        serving_input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(features)
        Estimator.export_savedmodel(FLAGS.servable_model_dir, serving_input_receiver_fn)


if __name__ == "__main__":
    tf.app.run()