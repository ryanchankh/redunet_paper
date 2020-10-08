import os
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import dataset
import utils


def get_n_each(X, y, n=1):
    classes = np.unique(y)
    _X, _y = [], []
    for c in classes:
        idx = y==c
        X_class = X[idx][:n]
        y_class = y[idx][:n]
        _X.append(X_class)
        _y.append(y_class)
    return np.vstack(_X), np.hstack(_y)

def translate1d(data, labels, stride=1):
    n_samples, _, n_dim = data.shape
    data_new, labels_new = [], []
    for i in range(n_samples):
        for r in range(0, n_dim, stride):
            data_new.append(np.roll(data[i], r, axis=1))
            labels_new.append(labels[i])
    data = np.stack(data_new)
    labels = np.array(labels_new)
    return data, labels

def translate2d(data, labels, stride=1):
    n_samples, _, H, W = data.shape
    data_new, labels_new = [], []
    for i in range(n_samples):
        for h in range(0, H, stride):
            for w in range(0, W, stride):
                data_new.append(np.roll(data[i], (h, w), axis=(1, 2)))
                labels_new.append(labels[i])
    data = np.stack(data_new)
    labels = np.array(labels_new)
    return data, labels

def shuffle(data, labels):
    num_samples = data.shape[0]
    idx = np.random.choice(np.arange(num_samples), num_samples, replace=False)
    return data[idx], labels[idx]

def filter_class(data, labels, classes, num_samples=None):
    if type(classes) == int:
        classes = np.arange(classes)
    data_filter = []
    labels_filter = []
    for _class in classes:
        idx = labels == _class
        data_filter.append(data[idx][:num_samples])
        labels_filter.append(labels[idx][:num_samples])
    data_new = np.vstack(data_filter)
    labels_new = np.unique(np.hstack(labels_filter), return_inverse=True)[1]
    return data_new, labels_new
    
def normalize(X, p=2):
    axes = tuple(np.arange(1, len(X.shape)).tolist())
    norm = np.linalg.norm(X.reshape(X.shape[0], -1), axis=1, ord=p)
    norm = np.clip(norm, 1e-8, np.inf)
    return X / np.expand_dims(norm, axes)

def batch_cov(V, bs):
    m = V.shape[0]
    return np.sum([np.einsum('ji...,jk...->ik...', V[i:i+bs], V[i:i+bs].conj(), optimize=True) \
                     for i in np.arange(0, m, bs)], axis=0)

def generate_kernel(mode, out_channels, in_channels, kernel_size):
    if mode == 'uniform':
        return np.random.rand(out_channels, in_channels, kernel_size)
    elif mode == 'gaussian':
        return np.random.normal(0, 1, size=(out_channels, in_channels, kernel_size))
    elif mode == 'ones':
        return np.ones(size=(out_channels, in_channels, kernel_size))

