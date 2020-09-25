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


def filter_class(data, labels, class_to_keep):
    data_filter = []
    labels_filter = []
    for _class in np.arange(class_to_keep):
        idx = labels == _class
        data_filter.append(data[idx])
        labels_filter.append(labels[idx])
    return np.vstack(data_filter), np.hstack(labels_filter)

def label_to_membership(targets, num_classes=None):
    """Generate a true membership matrix, and assign value to current Pi.

    Parameters:
        targets (np.ndarray): matrix with one hot labels

    Return:
        Pi: membership matirx, shape (num_classes, num_samples, num_samples)

    """
    targets = one_hot(targets, num_classes)
    num_samples, num_classes = targets.shape
    Pi = np.zeros(shape=(num_classes, num_samples, num_samples))
    for j in range(len(targets)):
        k = np.argmax(targets[j])
        Pi[k, j, j] = 1.
    return torch.tensor(Pi, dtype=torch.float)

def membership_to_label(membership):
    """Turn a membership matrix into a list of labels."""
    _, num_classes, num_samples, _ = membership.shape
    labels = np.zeros(num_samples)
    for i in range(num_samples):
        labels[i] = np.argmax(membership[:, i, i])
    return labels

def one_hot(y, num_classes):
    """Turn labels x into one hot vector of K classes. """
    label = torch.zeros(size=(y.shape[0], num_classes))
    for i in range(len(y)):
        label[i][y[i]] = 1.
    return label

def corrupt_labels(mode):
    if mode == "shuffle":
        from corrupt import shuffle_corrupt
        return shuffle_corrupt
    elif mode == "pairflip":
        from corrupt import noisify_pairflip
        return noisify_pairflip
    raise NameError("{} corruption mode not found.".format(mode))
    
def normalize(X, p=2):
    axes = tuple(np.arange(1, len(X.shape)).tolist())
    norm = np.linalg.norm(X.reshape(X.shape[0], -1), axis=1, ord=p)
    norm = np.clip(norm, 1e-8, np.inf)
    return X / np.expand_dims(norm, axes)

def batch_cov(V, bs):
    m = V.shape[0]
    return np.sum([np.einsum('ji...,jk...->ik...', V[i:i+bs], V[i:i+bs].conj(), optimize=True) \
                     for i in np.arange(0, m, bs)], axis=0)

def create_group(X, num_classes):
    return (np.vstack([X for _ in range(num_classes)]),
            np.repeat(np.arange(num_classes), X.shape[0]))