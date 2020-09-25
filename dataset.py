import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
import scipy.ndimage
from sklearn.preprocessing import normalize

from sklearn.model_selection import train_test_split
from sklearn.datasets import make_s_curve

import torch
from torch.utils.data import Dataset
from torchvision.datasets import MNIST


def generate_kernel(mode, out_channels, in_channels, kernel_size):
    if mode == 'random':
        return np.random.rand(out_channels, in_channels, kernel_size)
    elif mode == 'ones':
        return np.ones(size=(out_channels, in_channels, kernel_size))

def generate_wave(time, 
                  samples, 
                  mode, 
                  shuffle=False, 
                  augment=False,
                  channel_dim=True):
    if mode == 1: # noiseless 2 class differnet frequency
        x0 = np.random.uniform(low=0, high=10*np.pi, size=samples)
        x1 = np.linspace(x0, x0+2*np.pi, time).T
        y1 = np.sin(x1 * 1)
        y2 = np.sin(x1 * 3)
        data = np.vstack([y1, y2])
        labels = np.hstack([np.ones(samples) * 0, 
                            np.ones(samples) * 1]).astype(np.int32)
        num_classes = 2
    elif mode == 2: # noiseless 3 class differnet frequency
        x0 = np.random.uniform(low=0, high=10*np.pi, size=samples)
        x1 = np.linspace(x0, x0+2*np.pi, time).T
        y1 = np.sin(x1 * 1)
        y2 = np.sin(x1 * 3)
        y3 = np.sin(x1 * 7)
        data = np.vstack([y1, y2, y3])
        labels = np.hstack([np.ones(samples) * 0, 
                            np.ones(samples) * 1,
                            np.ones(samples) * 2]).astype(np.int32)
        num_classes = 3
    elif mode == 3: # noiseless 4 class differnet frequency
        x0 = np.random.uniform(low=0, high=10*np.pi, size=samples)
        x1 = np.linspace(x0, x0+2*np.pi, time).T
        y1 = np.sin(x1 * 1)
        y2 = np.sin(x1 * 3)
        y3 = np.sin(x1 * 7)
        y4 = np.sin(x1 * 11)
        data = np.vstack([y1, y2, y3, y4])
        labels = np.hstack([np.ones(samples) * 0, 
                            np.ones(samples) * 1,
                            np.ones(samples) * 2,
                            np.ones(samples) * 3]).astype(np.int32)
        num_classes = 4
    elif mode == 4: # noiseless hill-valley
        x0 = np.random.randint(0, time, samples)
        y0 = np.stack([norm.pdf(np.arange(time), loc=_x0, scale=5) for _x0 in x0])
        x1 = np.random.randint(0, time, samples)
        y1 = np.stack([-norm.pdf(np.arange(time), loc=_x0, scale=5) for _x0 in x0])
        data = np.vstack([y0, y1])
        labels = np.hstack([np.zeros(shape=samples), np.ones(shape=samples)])
        num_classes = 2
    elif mode == 5: # noiseless sign vs sin
        x0 = np.random.uniform(low=0, high=10*np.pi, size=samples)
        x1 = np.linspace(x0, x0+2*np.pi, time).T
        y1 = np.sin(x1 * 1)
        y2 = np.sign(np.sin(x1 * 1))
        data = np.vstack([y1, y2])
        labels = np.hstack([np.ones(samples) * 0, 
                            np.ones(samples) * 1]).astype(np.int32)
        num_classes = 2
    elif mode == 6: # noiseless 4 classes sign vs sin
        x0 = np.random.uniform(low=0, high=10*np.pi, size=samples)
        x1 = np.linspace(x0, x0+2*np.pi, time).T
        y1 = np.sin(x1 * 1)
        y2 = np.sign(np.sin(x1 * 1))
        y3 = np.sin(x1 * 3)
        y4 = np.sign(np.sin(x1 * 3))
        data = np.vstack([y1, y2, y3, y4])
        labels = np.hstack([np.ones(samples) * 0, 
                            np.ones(samples) * 1,
                            np.ones(samples) * 2,
                            np.ones(samples) * 3]).astype(np.int32)
        num_classes = 4
    elif mode == 7: # sign vs sin with noise var 0.1
        x0 = np.random.uniform(low=0, high=10*np.pi, size=samples)
        x1 = np.linspace(x0, x0+2*np.pi, time).T
        y1 = np.sin(x1 * 1) + np.random.normal(loc=0, scale=0.1, size=(samples, time))
        y2 = np.sign(np.sin(x1 * 1)) + np.random.normal(loc=0, scale=0.1, size=(samples, time))
        data = np.vstack([y1, y2])
        labels = np.hstack([np.ones(samples) * 0, 
                            np.ones(samples) * 1]).astype(np.int32)
        num_classes = 2
    elif mode == 8: # sign vs sin with noise var 0.2
        x0 = np.random.uniform(low=0, high=10*np.pi, size=samples)
        x1 = np.linspace(x0, x0+2*np.pi, time).T
        y1 = np.sin(x1 * 1) + np.random.normal(loc=0, scale=0.2, size=(samples, time))
        y2 = np.sign(np.sin(x1 * 1)) + np.random.normal(loc=0, scale=0.2, size=(samples, time))
        data = np.vstack([y1, y2])
        labels = np.hstack([np.ones(samples) * 0, 
                            np.ones(samples) * 1]).astype(np.int32)
        num_classes = 2
    else:
        raise NameError("Dataset not found.")
    if shuffle:
        idx = np.random.choice(np.arange(data.shape[0]), 
                               data.shape[0], 
                               replace=False)
        data, labels = data[idx], labels[idx]
        
    if augment:
        n_samples, n_dim = data.shape
        data_new, labels_new = [], []
        for i in range(n_samples):
            for r in range(n_dim):
                data_new.append(np.roll(data[i], r))
                labels_new.append(labels[i])
        data = np.stack(data_new).astype(np.float32)
        labels = np.array(labels_new).astype(np.int32)
    
    if channel_dim:
        data = np.expand_dims(data, 1)
    return data, labels, num_classes

def generate_scurve(samples, noise, num_classes):
    X, t = make_s_curve(samples, noise=noise)
    spaces = np.linspace(t.min(), t.max(), num_classes+1)
    y = np.empty(samples)
    for i in range(spaces.size-1):
        idx = ((spaces[i] <= t) * (spaces[i] <= t))
        y[idx] = i
    return X.astype(np.float32), y.astype(np.int32)

def generate_2d(data, noise, samples, shuffle=False):
    if data == 1:
        centers = [(1, 0), (0, 1)]
    else:
        raise NameErorr('data not found.')

    data = []
    targets = []
    for lbl, center in enumerate(centers):
        X = np.random.normal(loc=center, scale=noise, size=(samples, 2))
        y = np.repeat(lbl, samples).tolist()
        data.append(X)
        targets += y
    data = np.concatenate(data)
    data = normalize(data, axis=1)
    targets = np.array(targets)

    if shuffle:
        idx_arr = np.random.choice(np.arange(len(data)), len(data), replace=False)
        data, targets = data[idx_arr], targets[idx_arr]

    return data, targets, len(centers)

def generate_3d(data, noise, samples, shuffle=False):
    if data == 1: # 2 curves on sphere
        theta1 = np.random.uniform(np.pi / 4,  -np.pi / 4, samples)
        x1 = np.cos(theta1)
        y1 = np.sin(5*theta1)
        z1 = np.sin(5*theta1)*x1 + 1.0
        X1 = np.stack([x1, y1, z1]).T
        X1 = normalize(X1, axis=1)
        Y1 = np.ones(samples) * 0
        theta2 = np.random.uniform(np.pi / 4,  -np.pi / 4, samples)
        x2 = np.sin(2*theta2)
        y2 = np.cos(2*theta2) + 1.0
        z2 = np.cos(2*theta2) * x2
        X2 = np.stack([x2, y2, z2]).T
        X2 = normalize(X2, axis=1)
        Y2 = np.ones(samples) * 1

        data = np.vstack([X1, X2])
        targets = np.hstack([Y1, Y2])
        num_classes = 2

    elif data == 2:  # 3 polar 
        x1 = np.random.normal(loc=0, scale=noise, size=samples)
        y1 = np.random.normal(loc=0, scale=noise, size=samples)
        z1 = np.ones(samples)
        X1 = np.stack([x1, y1, z1]).T
        X1 = normalize(X1, axis=1)
        Y1 = np.ones(samples) * 0

        x2 = np.random.normal(loc=0, scale=noise, size=samples)
        y2 = np.ones(samples)
        z2 = np.random.normal(loc=0, scale=noise, size=samples)
        X2 = np.stack([x2, y2, z2]).T
        X2 = normalize(X2, axis=1)
        Y2 = np.ones(samples) * 1

        x3 = np.ones(samples)
        y3 = np.random.normal(loc=0, scale=noise, size=samples)
        z3 = np.random.normal(loc=0, scale=noise, size=samples)
        X3 = np.stack([x3, y3, z3]).T
        X3 = normalize(X3, axis=1)
        Y3 = np.ones(samples) * 2

        data = np.vstack([X1, X2, X3])
        targets = np.hstack([Y1, Y2, Y3])
        num_classes = 3
    else:
        raise NameError('Class not found.')

    if shuffle:
        idx_arr = np.random.choice(np.arange(len(data)), len(data), replace=False)
        data, targets = data[idx_arr], targets[idx_arr]

    return data.astype(np.float32), targets.astype(np.int32), num_classes


def load_MNIST_polar(root, samples, channels, time, train=True):
    dataset = MNIST(root, train=train, download=True)
    images = dataset.data[:samples].numpy()
    labels = dataset.targets[:samples].numpy()
    mid_pt = images.shape[1] // 2
    r = np.linspace(0, mid_pt, channels).astype(np.int32)
    angles = np.linspace(0, 360, time)

    polar_imgs = []
    for angle in angles:
        X_rot = scipy.ndimage.rotate(images, angle, axes=(1, 2), reshape=False)
        polar_imgs.append(X_rot[:, mid_pt, r])
    polar_imgs = np.stack(polar_imgs).transpose(1, 2, 0)
    return polar_imgs, labels

