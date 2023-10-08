import tensorflow as tf
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score
import pickle
import os
import random
# tf.compat.v1.enable_eager_execution()

def build_mask_matrix(seqlen, valid_len_list):
    '''
        (1) if a sequence of length 4 contains zero padding token (i.e., the valid length is 4),
            then the loss padding matrix looks like
                 [0., 1., 1., 1.],
                 [1., 0., 1., 1.],
                 [1., 1., 0., 1.],
                 [1., 1., 1., 0.]

        (2) if a sequence of length 4 contains 1 padding token (i.e., the valid length is 3),
            then the loss padding matrix looks like
                 [0., 1., 1., 0.],
                 [1., 0., 1., 0.],
                 [1., 1., 0., 0.],
                 [0., 0., 0., 0.]
    '''
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

def add_gaussian_noise(x_concat, std=0.01):
    noise = tf.random.normal(shape=tf.shape(x_concat), mean=0.0, stddev=std)
    x_noisy = x_concat + noise
    return x_noisy

def add_uniform_noise(x_concat, weight):
    random_noise = tf.random.uniform(shape=tf.shape(x_concat))
    # 生成与x_concat相同size的噪声, 填充均匀分布的随机数值
    x_concat += tf.sign(x_concat) * tf.nn.l2_normalize(random_noise, axis=-1) * weight
    # 控制噪声的两个条件，并生成新的view
    return x_concat

def contrastive_loss(score_matrix, margin, bsz):
    '''
       score_matrix: bsz x seqlen x seqlen
       input_ids: bsz x seqlen
    '''
    seqlen = score_matrix.get_shape().as_list()[1]

    gold_score = tf.linalg.diag_part(score_matrix)  # bsz x seqlen
    gold_score = tf.expand_dims(gold_score, -1)
    # assert gold_score.shape == tf.TensorShape([bsz, seqlen, 1])
    difference_matrix = gold_score - score_matrix

    # assert difference_matrix.shape == tf.TensorShape([bsz, seqlen, seqlen])
    loss_matrix = margin - difference_matrix  # bsz x seqlen x seqlen
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

def cts_loss_diff_experts(last_hidden_states, batch_size):
    '''
        last_hidden_states: bsz x seqlen x embed_dim
        logits: bsz x seqlen x vocab_size
        input_ids: bsz x seqlen
        labels: bsz x seqlen
    '''
    # compute cl loss
    norm_rep = last_hidden_states / tf.norm(last_hidden_states, axis=-1, keepdims=True)
    cosine_scores = tf.matmul(norm_rep, norm_rep, transpose_b=True)
    # assert cosine_scores.shape == tf.TensorShape([10240, 5, 5])
    cl_loss = contrastive_loss(score_matrix=cosine_scores, margin=0.5, bsz=batch_size)
    return cl_loss

def random_dropout_embedding(x_concat, keep_prob=0.95):
    # print(x_concat.get_shape().as_list())
    # print('&&&&&&&&&&&&&')
    # assert 1==2
    x_concat_drop = tf.nn.dropout(x_concat, keep_prob=keep_prob)
    return x_concat_drop


# tf.enable_eager_execution()
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

mi_values = None
mi_matrix = None
mask_indices = None
is_first_batch = True

def cfm_masking_F_L(x, mask_prob, labels):
    global is_first_batch, mi_values, mask_indices
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        x, labels = sess.run([x, labels])
    if is_first_batch:
        load_mi_values(x, labels)
        sorted_indices = np.argsort(mi_values)
        num_masked = round(mask_prob * x.shape[1])
        # 比较低的互信息indices
        mask_indices = sorted_indices[:num_masked]
        is_first_batch = False
    # 比较高的互信息indices
    # mask_indices = sorted_indices[-num_masked:]
    # 将对应的特征列置为0
    x_masked = np.copy(x)
    x_masked[:, mask_indices] = 0.0
    # 将NumPy数组转换为TensorFlow张量
    x_masked_tensor = tf.convert_to_tensor(x_masked, dtype=tf.float32)
    return x_masked_tensor

def load_mi_values(x, labels):
    global mi_values
    if os.path.isfile('/data/mi_values.pkl'):
        with open('/data/mi_values.pkl', 'rb') as file:
            mi_values = pickle.load(file)
    else:
        mi_values = mutual_info_regression(x, labels)
        with open('/data/mi_values.pkl', 'wb') as file:
            pickle.dump(mi_values, file)

def compute_similarity_matrix(x):
    num_features = x.shape[1]
    similarity_matrix = np.dot(x.T, x) / np.outer(np.linalg.norm(x, axis=0), np.linalg.norm(x, axis=0))
    similarity_matrix[np.isnan(similarity_matrix)] = 0  # 处理由于除以0导致的NaN值
    similarity_matrix[np.isinf(similarity_matrix)] = 0  # 处理由于除以0导致的Inf值
    similarity_matrix = np.triu(similarity_matrix) + np.triu(similarity_matrix, 1).T  # 计算上三角部分和对称部分
    assert similarity_matrix.shape == (num_features, num_features)
    return similarity_matrix

def load_mi_matrix(x):
    global mi_matrix
    if os.path.isfile('/data/mi_matrix_fullsize_tensor.pkl'):
        with open('/data/mi_matrix_fullsize_tensor.pkl', 'rb') as file:
            mi_matrix = pickle.load(file)
    else:
        # mi_matrix = compute_mutual_info_matrix(x)
        mi_matrix = compute_similarity_matrix(x)
        with open('/data/mi_matrix_fullsize_tensor.pkl', 'wb') as file:
            pickle.dump(mi_matrix, file)

def cfm_masking_F_F(x, mask_prob, num_features):
    global is_first_batch, mi_matrix
    # sees run x to numoy array
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        x = sess.run(x)
    # only load mi_matrix in first_batch and global it
    if is_first_batch:
        load_mi_matrix(x)
        is_first_batch = False

    selected_feature = np.random.choice(2688)  # 随机选择一个特征

    n = round(mask_prob * num_features)
    # 找到与选定特征互信息值最高的n个特征
    sorted_indices = np.argsort(mi_matrix[selected_feature])
    mask_indices = sorted_indices[-(n+1):-1]  # 取互信息值最高的n个特征

    # 将选中的特征及其互信息值高的n个特征进行mask
    x_masked = np.copy(x)
    x_masked[:, [selected_feature] + list(mask_indices)] = 0.0
    # np.savetxt('/data/x_masked.txt', x_masked, fmt='%.12f')
    # 将NumPy数组转换为TensorFlow张量
    x_masked_tensor = tf.convert_to_tensor(x_masked, dtype=tf.float32)
    return x_masked_tensor

def calculate_mutual_information_feat(feat_ids):
    if os.path.isfile('/data/mi_matrix_feat_ids.pkl'):
        with open('/data/mi_matrix_feat_ids.pkl', 'rb') as file:
            feat_mi = pickle.load(file)
    else:
        feat_ids = session_run(feat_ids)
        num_samples, num_features = feat_ids.shape
        feat_mi = np.zeros((num_features, num_features))
        for i in range(num_features):
            for j in range(i + 1, num_features):  # 只计算上三角部分的元素
                x = feat_ids[:, i]
                y = feat_ids[:, j]

                mi = mutual_info_regression(x.reshape(-1, 1), y)
                feat_mi[i, j] = mi
                feat_mi[j, i] = mi  # 填充对称位置的元素
        with open('/data/mi_matrix_feat_ids.pkl', 'wb') as file:
            pickle.dump(feat_mi, file)
    return feat_mi

def session_run(content):
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        content = sess.run(content)
    return content

# def cfm_masking_F_F_feat(x, feat_ids, mask_prob, num_features):
#     x = session_run(x)
#     frequency_matrix = calculate_frequency_matrix(feat_ids)
#     n = round(mask_prob * num_features)
#     mask_indices = generate_mask_indices(frequency_matrix, n)
#     x_masked = np.copy(x)
#
#     for i in range(mask_indices.shape[0]):
#         mask_index = mask_indices[i]
#         for j in mask_index:
#             x_masked[i, j * 32: (j + 1) * 32] = 0.0
#     x_masked_tensor = tf.convert_to_tensor(x_masked, dtype=tf.float32)
#     return x_masked_tensor, mask_indices

def calculate_frequency_matrix(feat_ids):
    feat_ids = session_run(feat_ids)
    # 计算feat_ids每一列的值出现的频率
    num_rows, num_cols = feat_ids.shape
    frequency_matrix = np.zeros((num_rows, num_cols))

    for i in range(num_cols):
        column_values, counts = np.unique(feat_ids[:, i], return_counts=True)
        column_frequency = counts / num_rows
        column_dict = dict(zip(column_values, column_frequency))
        for j in range(num_rows):
            frequency_matrix[j, i] = column_dict[feat_ids[j, i]]
    return frequency_matrix

def cfm_masking_F_F_feat(x, feat_ids, mask_prob):
    x = session_run(x)
    frequency_matrix = calculate_frequency_matrix(feat_ids)
    x_masked = np.copy(x)
    for i in range(frequency_matrix.shape[0]):
        for j in range(frequency_matrix.shape[1]):
            mask_probability = frequency_matrix[i, j] # 获取对应位置的概率值
            if random.random() < mask_probability * mask_prob: # 根据概率进行mask
                x_masked[i, j * 32: (j + 1) * 32] = 0.0
    x_masked_tensor = tf.convert_to_tensor(x_masked, dtype=tf.float32)
    return x_masked_tensor, mask_indices


def cfm_masking_F_F_feat_mi(x, feat_ids, mask_prob, num_features):
    x = session_run(x)
    mi_matrix = calculate_mutual_information_feat(feat_ids)
    selected_feature = np.random.choice(num_features)  # 随机选择一个特征
    n = round(mask_prob * num_features)
    # 找到与选定特征互信息值最高的n个特征
    sorted_indices = np.argsort(mi_matrix[selected_feature])
    mask_indices = sorted_indices[-(n+1):-1]  # 取互信息值最高的n个特征
    # 将选中的特征及其互信息值高的n个特征进行mask
    x_masked = np.copy(x)
    for i in mask_indices:
        x_masked[:, range(i*32, (i+1)*32)] = 0.0
    x_masked[:, range(selected_feature*32, (selected_feature+1)*32)] = 0.0
    x_masked_tensor = tf.convert_to_tensor(x_masked, dtype=tf.float32)
    return x_masked_tensor, mask_indices

def calculate_mutual_information_feat2(feat_ids):
    feat_ids = session_run(feat_ids)
    num_samples, num_features = feat_ids.shape
    feat_mi = np.zeros((num_features, num_features))
    for i in range(num_features):
        for j in range(i + 1, num_features):  # 只计算上三角部分的元素
            x = feat_ids[:, i]
            y = feat_ids[:, j]

            mi = mutual_info_regression(x.reshape(-1, 1), y)
            feat_mi[i, j] = mi
            feat_mi[j, i] = mi  # 填充对称位置的元素
    return feat_mi

def cfm_masking_F_F_feat_mi2(x, feat_ids, mask_prob, num_features):
    x = session_run(x)
    mi_matrix = calculate_mutual_information_feat2(feat_ids)
    selected_feature = np.random.choice(num_features)  # 随机选择一个特征
    n = round(mask_prob * num_features)
    # 找到与选定特征互信息值最高的n个特征
    sorted_indices = np.argsort(mi_matrix[selected_feature])
    mask_indices = sorted_indices[-(n+1):-1]  # 取互信息值最高的n个特征
    # 将选中的特征及其互信息值高的n个特征进行mask
    x_masked = np.copy(x)
    for i in mask_indices:
        x_masked[:, range(i*32, (i+1)*32)] = 0.0
    x_masked[:, range(selected_feature*32, (selected_feature+1)*32)] = 0.0
    x_masked_tensor = tf.convert_to_tensor(x_masked, dtype=tf.float32)
    return x_masked_tensor, mask_indices

# def compute_mutual_info_matrix(x):
#     num_features = x.shape[1]
#     mi_matrix = tf.Variable(tf.zeros((num_features, num_features)))
#     for i in range(num_features):
#         y = x[:, i]
#         mi_values = compute_mutual_info(x, y)
#         # print(session_run(mi_values))
#         # print('***************')
#         mi_matrix[i, :].assign(mi_values)
#         mi_matrix[:, i].assign(mi_values)
#     return mi_matrix

# def compute_similarity_matrix(x):
#     num_features = x.shape[1]
#     similarity_matrix = np.dot(x.T, x)  # 计算向量内积
#
#     norms = np.linalg.norm(x, axis=0)  # 计算每列向量的模长
#     norms_matrix = np.outer(norms, norms)  # 通过广播创建模长矩阵
#     similarity_matrix /= norms_matrix  # 归一化
#
#     return similarity_matrix


# def compute_mutual_info_matrix(x):
#     # 计算x的概率分布
#     x_prob = tf.nn.softmax(x, axis=0)
#     # 计算互信息
#     n_features = x.shape[1]
#     mi = tf.zeros((n_features, n_features), dtype=tf.float32)
#     updates = []
#     for i in range(n_features):
#         for j in range(i + 1, n_features):
#             mi_ij = tf.reduce_sum(x_prob[:, i] * tf.math.log(x_prob[:, i] / (x_prob[:, i] + x_prob[:, j]) + 1e-8))
#             mi_ji = tf.reduce_sum(x_prob[:, j] * tf.math.log(x_prob[:, j] / (x_prob[:, j] + x_prob[:, i]) + 1e-8))
#             print(session_run([mi_ji, mi_ij]))
#             print('&&&&&&&&&&&&&&&&&&&')
#             updates.append([i, j, mi_ij])
#             updates.append([j, i, mi_ji])
#     mi = tf.tensor_scatter_nd_update(mi, indices=updates, updates=tf.reshape(mi, [-1]))
#     return mi

# def cfm_masking_F_F(x, mask_prob, num_features):
#     global is_first_batch, mi_matrix
#     init_op = tf.global_variables_initializer()
#     with tf.Session() as sess:
#         sess.run(init_op)
#         x = sess.run(x)
#     if is_first_batch:
#         # load_mi_matrix(x)
#         load_mi_matrix(x)
#         is_first_batch = False
#
#     selected_feature = tf.random.uniform(shape=(), maxval=num_features, dtype=tf.int32)
#     n = tf.round(mask_prob * num_features)
#     sorted_indices = tf.argsort(mi_matrix[selected_feature])
#     mask_indices = sorted_indices[-(n+1):-1]
#     x_masked = tf.identity(x)
#     x_masked = tf.tensor_scatter_nd_update(x_masked, tf.concat([[selected_feature]], mask_indices, axis=0), tf.zeros_like(mask_indices, dtype=tf.float32))
#     return x_masked

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


