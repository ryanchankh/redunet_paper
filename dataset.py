import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.ndimage
import pandas as pd
from pandas.api.types import CategoricalDtype
from sklearn.datasets import load_iris
from sklearn.datasets import make_s_curve
from sklearn.model_selection import train_test_split
import torch

import functionals as F

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
    elif mode == 6: # 4 classes sign vs sin with 0.1
        x0 = np.random.uniform(low=0, high=10*np.pi, size=samples)
        x1 = np.linspace(x0, x0+2*np.pi, time).T
        y1 = np.sin(x1 * 1) + np.random.normal(loc=0, scale=0.1, size=(samples, time))
        y2 = np.sign(np.sin(x1 * 1)) + np.random.normal(loc=0, scale=0.1, size=(samples, time))
        y3 = np.sin(x1 * 3) + + np.random.normal(loc=0, scale=0.1, size=(samples, time))
        y4 = np.sign(np.sin(x1 * 3)) + + np.random.normal(loc=0, scale=0.1, size=(samples, time))
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
    elif mode == 9:
        x0 = np.random.uniform(low=0, high=10*np.pi, size=samples)
        x1 = np.linspace(x0, x0+2*np.pi, time).T
        f1 = np.sin(x1) + np.random.normal(loc=0, scale=0.1, size=(samples, time))
        f2 = np.e**(np.sin(x1)) + np.random.normal(loc=0, scale=0.1, size=(samples, time))
        f3 = np.sign(np.sin(x1)) + np.random.normal(loc=0, scale=0.1, size=(samples, time))
        data = np.vstack([f1, f2, f3])
        labels = np.hstack([np.ones(samples) * 0, 
                            np.ones(samples) * 1,
                            np.ones(samples) * 2]).astype(np.int32)
        num_classes = 3
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
    elif data == 2:
        centers = [(np.cos(np.pi/3), np.sin(np.pi/3)), (1 ,0)]
    elif data == 3:
        centers = [(np.cos(np.pi/4), np.sin(np.pi/4)), (1 ,0)]
    elif data == 4:
        centers = [(np.cos(3*np.pi/4), np.sin(3*np.pi/4)), (1 ,0)]
    elif data == 5:
        centers = [(np.cos(2*np.pi/3), np.sin(2*np.pi/3)), (1 ,0)]
    elif data == 6:
        centers = [(np.cos(3*np.pi/4), np.sin(3*np.pi/4)), (np.cos(4*np.pi/3), np.sin(4*np.pi/3)), (1 ,0)]
    elif data == 7:
        centers = [(np.cos(3*np.pi/4), np.sin(3*np.pi/4)), 
                   (np.cos(4*np.pi/3), np.sin(4*np.pi/3)), 
                   (np.cos(np.pi/4), np.sin(np.pi/4))]
    elif data == 8:
        centers = [(np.cos(np.pi/6), np.sin(np.pi/6)), 
                   (np.cos(np.pi/2), np.sin(np.pi/2)), 
                   (np.cos(3*np.pi/4), np.sin(3*np.pi/4)),
                   (np.cos(5*np.pi/4), np.sin(5*np.pi/4)),
                   (np.cos(7*np.pi/4), np.sin(7*np.pi/4)),
                   (np.cos(3*np.pi/2), np.sin(3*np.pi/2))]
    else:
        raise NameError('data not found.')

    data = []
    targets = []
    for lbl, center in enumerate(centers):
        X = np.random.normal(loc=center, scale=noise, size=(samples, 2))
        y = np.repeat(lbl, samples).tolist()
        data.append(X)
        targets += y
    data = np.concatenate(data)
    data = F.normalize(data)
    targets = np.array(targets)

    if shuffle:
        idx_arr = np.random.choice(np.arange(len(data)), len(data), replace=False)
        data, targets = data[idx_arr], targets[idx_arr]

    return data, targets, len(centers)

def generate_3d(data, noise, samples, shuffle=False):
    if data == 1:
        centers = [(1, 0, 0), 
                   (0, 1, 0), 
                   (0, 0, 1)]
    elif data == 2:
        centers = [(np.cos(np.pi/4), np.sin(np.pi/4), 1),
                   (np.cos(2*np.pi/3), np.sin(2*np.pi/3), 1),
                   (np.cos(np.pi), np.sin(np.pi), 1)]
    elif data == 3:
        centers = [(np.cos(np.pi/4), np.sin(np.pi/4), 1),
                   (np.cos(2*np.pi/3), np.sin(2*np.pi/3), 1),
                   (np.cos(5*np.pi/6), np.cos(5*np.pi/6), 1)]
    else:
        raise NameError('Data not found.')

    X, Y = [], []
    for c, center in enumerate(centers):
        _X = np.random.normal(center, scale=(noise, noise, noise), size=(samples, 3))
        _Y = np.ones(samples, dtype=np.int32) * c
        X.append(_X)
        Y.append(_Y)
    X = F.normalize(np.vstack(X))
    Y = np.hstack(Y)
    
    if shuffle:
        idx_arr = np.random.choice(np.arange(len(X)), len(X), replace=False)
        X, Y = X[idx_arr], Y[idx_arr]

    return X.astype(np.float32), Y.astype(np.int32), 3


def load_MNIST(root, train=True):
    from torchvision.datasets import MNIST
    dataset = MNIST(root, train=train, download=True)
    images = dataset.data.numpy()
    labels = dataset.targets.numpy()
    return images, labels

def load_Iris(test_size=0.3, seed=42):
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True,
                                                        test_size=test_size,
                                                        random_state=seed)
    X_train = F.normalize(X_train)
    X_test = F.normalize(X_test)
    num_classes = 3
    return X_train, y_train, X_test, y_test, num_classes

def load_Mice(root, test_size=0.3, seed=42):
    df_data = pd.read_csv('./data/mice/data.csv')
    df_data = pd.read_csv(root)
    df_data = df_data.fillna(df_data.mean())
    df_data['class'] = df_data['class'].astype('category').cat.codes
    X = df_data.to_numpy()[:, 1:78].astype(np.float32)
    X = F.normalize(X)
    y = df_data['class'].to_numpy().astype(np.int32)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)
    num_classes = 8
    return X_train, y_train, X_test, y_test, num_classes


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

def load_MNIST_polar(root, samples, channels, time, train=True): #TODO
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
