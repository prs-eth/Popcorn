'''
Utilities to load and save data
'''

import json
import pickle

def save_json(obj, file, indent=4, sort_keys=True):
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

def load_json(file):
    with open(file, 'r') as f:
        a = json.load(f)
    return a

def save_pkl(file, path):
    with open(path, 'wb') as handle:
        pickle.dump(file, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_pkl(path):
    with open(path, 'rb') as handle:
        return pickle.load(handle)