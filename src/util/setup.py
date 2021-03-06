from util import clean_data as cd
import os
import csv
import numpy as np

def get_data(METADATA_PATH, PREFIX, POSTFIX):
    md = get_metadata(METADATA_PATH, ['FILE_ID', 'DX_GROUP'])
    good_md = filter_file_names(md, PREFIX, POSTFIX)
    n = len(good_md) # number of subjects
    f_names = np.array(map(lambda x: x['FILE_ID'], good_md))
    dx_groups = np.array(map(lambda x: int(x['DX_GROUP']), good_md))
    p = np.random.permutation(n)
    f_names = f_names[p]
    dx_groups = dx_groups[p]
    data = map(lambda p: np.loadtxt(p), f_names)
    return data, dx_groups, n

def get_data2(MDP1, PRE1, POST1, MDP2, PRE2, POST2):
    md1 = get_metadata(MDP1, ['FILE_ID', 'DX_GROUP'])
    good_md1 = filter_file_names2(md1, PRE1, POST1, PRE2, POST2)
    md2 = get_metadata(MDP2, ['FILE_ID', 'DX_GROUP'])
    good_md2 = filter_file_names2(md2, PRE2, POST2, PRE1, POST1)
    n = len(good_md1) # number of subjects
    assert len(good_md1) == len(good_md2)
    f_names1 = np.array(map(lambda x: x['FILE_ID'], good_md1))
    dx_groups1 = np.array(map(lambda x: int(x['DX_GROUP']), good_md1))
    f_names2 = np.array(map(lambda x: x['FILE_ID'], good_md2))
    dx_groups2 = np.array(map(lambda x: int(x['DX_GROUP']), good_md2))
    p = np.random.permutation(n)
    f_names1 = f_names1[p]
    dx_groups1 = dx_groups1[p]
    f_names2 = f_names2[p]
    dx_groups2 = dx_groups2[p]
    data1 = map(lambda p: np.loadtxt(p), f_names1)
    data2 = map(lambda p: np.loadtxt(p), f_names2)
    assert np.sum(dx_groups1 - dx_groups2) == 0
    return data1, data2, dx_groups1, dx_groups2, n

def split_data(data, labels, n_train, n_tune, n_test):
    D = {}
    L = {}
    D['train'] = split_groups(data[:n_train],
                                 labels[:n_train])
    L['train'] = labels[:n_train]
    D['tune'] = data[n_train:n_train+n_tune]
    L['tune'] = labels[n_train:n_train+n_tune]
    D['test'] = data[-n_test:]
    L['test'] = labels[-n_test:]
    return D, L

def split_groups(data, dx_groups):
    data_a = [i for (i, v) in zip(data, dx_groups == 1) if v]
    data_c = [i for (i, v) in zip(data, dx_groups == 2) if v]
    return {'a': data_a, 'c': data_c}

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

def filter_file_names2(meta_data, pre1, post1, pre2, post2):
    good_md = filter(lambda x: file_exists(pre1 + x['FILE_ID'] + post1) and file_exists(pre2 + x['FILE_ID'] + post2), meta_data)
    for md in good_md:
        md['FILE_ID'] = pre1 + md['FILE_ID'] + post1
    return good_md

def file_exists(f_name):
    return os.path.isfile(f_name)
