# clean_data.py
# Author: Ryan Marren
# Date: November 2016

import csv
import numpy as np
import os

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
    return f_names

def file_exists(f_name):
    return os.path.isfile(f_name)

def get_data(f_names):   
    return map(lambda p: np.loadtxt(p), f_names)

def run_function(fnc, data):   
    return map(lambda d: fnc(d), data)
