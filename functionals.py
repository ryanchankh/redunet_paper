import os
from tqdm import tqdm

import cv2
import numpy as np
import scipy
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

def translate1d(data, labels, n=None, stride=1):
    n_samples, _, n_dim = data.shape
    data_new = []
    if n is None:
        shifts = np.arange(0, n_dim, stride)
    else:
        shifts = np.arange(-n*stride, (n+1)*stride, stride)
    for r in shifts:
        data_new.append(np.roll(data, r, axis=2))
    return (np.vstack(data_new), 
            np.tile(labels, len(shifts)))

def translate2d(data, labels, n=None, stride=1):
    n_samples, _, H, W = data.shape
    data_new = []
    if n is None:
        vshifts = np.arange(0, H, stride)
        hshifts = np.arange(0, W, stride)
    else:
        hshifts = np.arange(-n*stride, (n+1)*stride, stride)
        vshifts = np.arange(-n*stride, (n+1)*stride, stride)
    for h in vshifts:
        for w in hshifts:
            data_new.append(np.roll(data, (h, w), axis=(2, 3)))
    return (np.vstack(data_new),
            np.tile(labels, len(vshifts)*len(hshifts)))

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

def convert2polar(images, channels, timesteps):
    mid_pt = images.shape[1] // 2
    r = np.linspace(0, mid_pt, channels).astype(np.int32)
    angles = np.linspace(0, 360, timesteps)
    polar_imgs = []
    for angle in angles:
        X_rot = scipy.ndimage.rotate(images, angle, axes=(1, 2), reshape=False)
        polar_imgs.append(X_rot[:, mid_pt, r])
    polar_imgs = np.stack(polar_imgs).transpose(1, 2, 0)
    return polar_imgs

