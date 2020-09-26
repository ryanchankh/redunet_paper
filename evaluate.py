import argparse
import os

import torch
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD

import dataset
# from arch import FullyConnected
import dataset
import train_func as tf
import utils


def svm(train_features, train_labels, test_features, test_labels):
    svm = LinearSVC(verbose=0, random_state=10)
    svm.fit(train_features, train_labels)
    acc_train = svm.score(train_features, train_labels)
    acc_test = svm.score(test_features, test_labels)
    print("SVM: {}".format(acc_test))
    return acc_train, acc_test

def knn(train_features, train_labels, test_features, test_labels, k=5):
    """Perform k-Nearest Neighbor classification using cosine similaristy as metric.
    Options:
        k (int): top k features for kNN
    
    """
    sim_mat = train_features @ test_features.T
    topk = torch.from_numpy(sim_mat).topk(k=k, dim=0)
    topk_pred = train_labels[topk.indices]
    test_pred = torch.tensor(topk_pred).mode(0).values.detach()
    acc = compute_accuracy(test_pred.numpy(), test_labels)
    print("kNN: {}".format(acc))
    return acc

def nearsub(train_features, train_labels, test_features, test_labels, classes, n_comp=10):
    """Perform nearest subspace classification.
    
    Options:
        n_comp (int): number of components for PCA or SVD
    
    """
    scores_pca = []
    scores_svd = []
    if type(classes) == int:
        classes = np.arange(classes)
    features_sort, _ = utils.sort_dataset(train_features, train_labels, 
                                          classes=classes, stack=False)           
    fd = features_sort[0].shape[1]
    if n_comp >= fd:
        n_comp = fd - 1
    for j in np.arange(len(classes)):
        pca = PCA(n_components=n_comp).fit(features_sort[j]) 
        pca_subspace = pca.components_.T
        mean = np.mean(features_sort[j], axis=0)
        pca_j = (np.eye(fd) - pca_subspace @ pca_subspace.T) \
                        @ (test_features - mean).T
        score_pca_j = np.linalg.norm(pca_j, ord=2, axis=0)

        svd = TruncatedSVD(n_components=n_comp).fit(features_sort[j])
        svd_subspace = svd.components_.T
        svd_j = (np.eye(fd) - svd_subspace @ svd_subspace.T) \
                        @ (test_features).T
        score_svd_j = np.linalg.norm(svd_j, ord=2, axis=0)
        
        scores_pca.append(score_pca_j)
        scores_svd.append(score_svd_j)
    test_predict_pca = np.argmin(scores_pca, axis=0)
    test_predict_svd = np.argmin(scores_svd, axis=0)
    acc_pca = compute_accuracy(test_predict_pca, test_labels)
    acc_svd = compute_accuracy(test_predict_svd, test_labels)
    print('PCA: {}'.format(acc_pca))
    print('SVD: {}'.format(acc_svd))
    return acc_svd

def nearsub_group(train_features, train_labels, test_features_group, test_labels, num_classes, n_comp=10):
    train_features_sort, _ = utils.sort_dataset(train_features, train_labels, 
                                          num_classes=num_classes, stack=False)           
    fd = train_features_sort[0].shape[1]
    if n_comp >= fd:
        n_comp = fd - 1

    class_subspaces = []
    for j in range(num_classes):
        svd = TruncatedSVD(n_components=n_comp).fit(train_features_sort[j])
        svd_subspace = svd.components_.T
        class_subspace = np.eye(fd) - svd_subspace @ svd_subspace.T
        class_subspaces.append(class_subspace)
    class_subspaces = np.stack(class_subspaces)
    
    class_scores = []
    class_features = []
    for j, i in enumerate(range(0, test_features_group.shape[0], test_labels.size)):
        test_features_j = test_features_group[i:i+test_labels.size]
        proj_j = class_subspaces[j] @ test_features_j.T
        score_j = np.linalg.norm(proj_j, ord=2, axis=0)
        class_scores.append(score_j)
        class_features.append(test_features_j)
    pred_labels = np.argmin(class_scores, axis=0)
    class_features = np.stack(class_features)

    pred_features = np.vstack([class_features[j][i] for i, j in enumerate(pred_labels)])
    test_features = np.vstack([class_features[j][i] for i, j in enumerate(test_labels)])
    acc = compute_accuracy(pred_labels, test_labels)
    print('nearsub: {}'.format(acc))
    return acc, pred_features, pred_labels, test_features

def compute_accuracy(y_pred, y_true):
    """Compute accuracy by counting correct classification. """
    assert y_pred.shape == y_true.shape
    return 1 - np.count_nonzero(y_pred - y_true) / y_true.size
