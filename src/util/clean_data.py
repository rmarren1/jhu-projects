# clean_data.py
# Author: Ryan Marren
# Date: November 2016

import csv
import numpy as np
import os
from sklearn.cluster import KMeans

def get_metadata(metadata_path, args):
    patient_list = []
    if 'FILE_ID' not in args:
        args.append(FILE_ID)
    with open(metadata_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            patient_list.append(row)
    num_patients = len(patient_list)
    meta_data = [{} for i in range(num_patients)]
    for keyword in args:
        for p in range(num_patients):
            meta_data[p][keyword] = patient_list[p][keyword]
    return meta_data

def get_filenames(meta_data, pre, post):
    f_names = map(lambda x: pre + x['FILE_ID'] + post, meta_data)
    f_names = filter(lambda x: file_exists(x), f_names)
    return f_names, inv_map

def filter_file_names(meta_data, pre, post):
    good_md = filter(lambda x: file_exists(pre + x['FILE_ID'] + post), meta_data)
    for md in good_md:
        md['FILE_ID'] = pre + md['FILE_ID'] + post
    return good_md


def file_exists(f_name):
    return os.path.isfile(f_name)

def get_data(f_names):   
    return map(lambda p: np.loadtxt(p), f_names)

def run_function(fnc, data):
    n = len(data)
    result = []
    checkpoints = np.append(np.arange(0, n, 100), [np.inf])
    for d in xrange(len(data)):
        result.append(fnc(data[d]))
        if d == checkpoints[0]:
            checkpoints = checkpoints[1:]
            print "Processed %04d of %04d brains." % (d, len(data))
    return result

def split_groups(data, dx_groups):
    data_a = [i for (i, v) in zip(data, dx_groups == 1) if v]
    data_c = [i for (i, v) in zip(data, dx_groups == 2) if v]
    return data_a, data_c

def concat_group(data):
    return np.hstack(data)

def get_kmeans_encoder(data, k):
    kmeans = KMeans(k).fit(data.T)
    return kmeans

def encode_samples(data, encoder, kind):
    return map(lambda x: kind + str(x), encoder.predict(data.T))

def get_dict(k, kind):
    return map(lambda x: kind + str(x), range(k))

def get_word_map(dict):
    n = len(dict)
    m = {}
    for i, symbol in enumerate(dict):
        arr = np.zeros(n)
        np.put(arr, i, 1)
        m[symbol] = arr
    return m

def context(words, h, side, mapping):
    words = np.array(words)
    s = 1
    if side == 'L':
        s = -1
    elif side == 'R':
        s = 1
    context = []
    for i in range(1, h + 1):
        context.append(np.roll(words, i * s))
    mats = map(lambda x: to_sparse_matrix(x, mapping), context)
    return np.vstack(mats)

def to_sparse_matrix(words, mapping):
    return np.array(map(lambda x: mapping[x], words)).T

def CCA(Z, W, k):
    Z = Z - np.mean(Z, axis=1).reshape(-1, 1)
    W = W - np.mean(W, axis=1).reshape(-1, 1)
    C_zz = Z.dot(Z.T)
    C_zw = Z.dot(W.T)
    C_ww = W.dot(W.T)
    C_wz = W.dot(Z.T)
    reg_z = np.eye(C_zz.shape[0]) * .05
    reg_w = np.eye(C_ww.shape[0]) * .05
    M_z = np.linalg.inv(C_zz + reg_z).dot(C_zw).dot(np.linalg.inv(C_ww + reg_w)).dot(C_wz)
    M_w = np.linalg.inv(C_ww + reg_w).dot(C_wz).dot(np.linalg.inv(C_zz + reg_z)).dot(C_zw)
    U_z, _, _ = np.linalg.svd(M_z)
    phi_z = U_z[:, :k]
    U_w, _, _ = np.linalg.svd(M_w)
    phi_w = U_w[:, :k]
    return (phi_z, phi_w)

def get_views(data, encoder, mapping, h):
    words = encode_samples(data, encoder, 'a')
    W = to_sparse_matrix(words, mapping)
    L = context(words, h, 'L', mapping)
    R = context(words, h, 'R', mapping)
    return W, L, R

def train(data, params):
    encoder = get_kmeans_encoder(data, params['k'])
    words = encode_samples(data, encoder, 'a')
    dictionary = get_dict(params['k'], 'a')
    mapping = get_word_map(dictionary)
    W, L, R = get_views(data, encoder, mapping, params['h'])
    phi_l, phi_r = CCA(L, R, params['d_cont'])
    S = np.vstack([phi_l.T.dot(L), phi_r.T.dot(R)])
    phi_s, phi_w = CCA(S, W, params['d_words'])
    model = {
        'phi_w' : phi_w,
        'phi_l' : phi_l,
        'phi_r' : phi_r,
        'encoder' : encoder,
        'mapping' : mapping,
        'params' : params
    }
    return model

def embed(data, M):
    W, L, R = get_views(data,
                        M['encoder'],
                        M['mapping'],
                        M['params']['h'])
    proj_w = M['phi_w'].T.dot(W)
    proj_l = M['phi_l'].T.dot(L)
    proj_r = M['phi_r'].T.dot(R)
    emb = np.vstack([proj_w, proj_l, proj_r])
    return emb


def correl_test(Z, W, phi_z, phi_w):
    Z = Z - np.mean(Z, axis=1).reshape(-1, 1)
    W = W - np.mean(W, axis=1).reshape(-1, 1)
    num = phi_z.T.dot(Z).dot(W.T).dot(phi_w)
    var_z = phi_z.T.dot(Z).dot(Z.T).dot(phi_z)
    var_w = phi_w.T.dot(W).dot(W.T).dot(phi_w)
    return num / np.sqrt(var_z * var_w)
