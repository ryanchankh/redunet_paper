import os
import logging
import json
import numpy as np
import pandas as pd
import torch

from torch.nn.functional import normalize


def sort_dataset(data, labels, classes, stack=False):
    """Sort dataset based on classes.
    
    Parameters:
        data (np.ndarray): data array
        labels (np.ndarray): one dimensional array of class labels
        classes (int): number of classes
        stack (bol): combine sorted data into one numpy array
    
    Return:
        sorted data (np.ndarray), sorted_labels (np.ndarray)

    """
    if type(classes) == int:
        classes = np.arange(classes)
    sorted_data = []
    sorted_labels = []
    for c in classes:
        idx = (labels == c)
        data_c = data[idx]
        labels_c = labels[idx]
        sorted_data.append(data_c)
        sorted_labels.append(labels_c)
    if stack:
        sorted_data = np.vstack(sorted_data)
        sorted_labels = np.hstack(sorted_labels)
    return sorted_data, sorted_labels

def save_params(model_dir, params, name='params.json'):
    """Save params to a .json file. Params is a dictionary of parameters."""
    path = os.path.join(model_dir, name)
    with open(path, 'w') as f:
        json.dump(params, f, indent=2, sort_keys=True)

def load_params(model_dir):
    """Load params.json file in model directory and return dictionary."""
    _path = os.path.join(model_dir, "params.json")
    with open(_path, 'r') as f:
        _dict = json.load(f)
    return _dict

def create_csv(model_dir, filename, headers):
    """Create .csv file with filename in model_dir, with headers as the first line 
    of the csv. """
    csv_path = os.path.join(model_dir, filename)
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(csv_path, 'w+') as f:
        f.write(','.join(map(str, headers)))
    return csv_path

def save_loss(loss_dict, model_dir, name):
    save_dir = os.path.join(model_dir, "loss")
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, "{}.csv".format(name))
    pd.DataFrame(loss_dict).to_csv(file_path)

def save_features(model_dir, name, features, labels, layer=None):
    save_dir = os.path.join(model_dir, "features")
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir, f"{name}_features.npy"), features)
    np.save(os.path.join(save_dir, f"{name}_labels.npy"), labels)
