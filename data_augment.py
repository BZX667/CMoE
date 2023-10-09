import tensorflow as tf
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.metrics import mutual_info_score
import pickle
import os
import random

mi_values = None
mi_matrix = None
mask_indices = None
is_first_batch = True


def session_run(content):
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        content = sess.run(content)
    return content


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


def random_dropout_embedding(x_concat, keep_prob=0.95):
    x_concat_drop = tf.nn.dropout(x_concat, keep_prob=keep_prob)
    return x_concat_drop


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
    mask_indices = sorted_indices[-(n + 1):-1]  # 取互信息值最高的n个特征

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
            mask_probability = frequency_matrix[i, j]  # 获取对应位置的概率值
            if random.random() < mask_probability * mask_prob:  # 根据概率进行mask
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
    mask_indices = sorted_indices[-(n + 1):-1]  # 取互信息值最高的n个特征
    # 将选中的特征及其互信息值高的n个特征进行mask
    x_masked = np.copy(x)
    for i in mask_indices:
        x_masked[:, range(i * 32, (i + 1) * 32)] = 0.0
    x_masked[:, range(selected_feature * 32, (selected_feature + 1) * 32)] = 0.0
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
    mask_indices = sorted_indices[-(n + 1):-1]  # 取互信息值最高的n个特征
    # 将选中的特征及其互信息值高的n个特征进行mask
    x_masked = np.copy(x)
    for i in mask_indices:
        x_masked[:, range(i * 32, (i + 1) * 32)] = 0.0
    x_masked[:, range(selected_feature * 32, (selected_feature + 1) * 32)] = 0.0
    x_masked_tensor = tf.convert_to_tensor(x_masked, dtype=tf.float32)
    return x_masked_tensor, mask_indices