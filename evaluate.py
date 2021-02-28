import argparse
import os

import torch
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import dataset
import functionals as F
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

def nearsub(train_features, train_labels, test_features, test_labels, n_comp=10):
    """Perform nearest subspace classification.
    
    Options:
        n_comp (int): number of components for PCA or SVD
    
    """
    scores_svd = []
    classes = np.unique(test_labels)
    features_sort, _ = utils.sort_dataset(train_features, train_labels, 
                                          classes=classes, stack=False)           
    fd = features_sort[0].shape[1]
    if n_comp >= fd:
        n_comp = fd - 1
    for j in np.arange(len(classes)):
        svd = TruncatedSVD(n_components=n_comp).fit(features_sort[j])
        svd_subspace = svd.components_.T
        svd_j = (np.eye(fd) - svd_subspace @ svd_subspace.T) \
                        @ (test_features).T
        score_svd_j = np.linalg.norm(svd_j, ord=2, axis=0)
        scores_svd.append(score_svd_j)
    test_predict_svd = np.argmin(scores_svd, axis=0)
    acc_svd = compute_accuracy(classes[test_predict_svd], test_labels)
    print('SVD: {}'.format(acc_svd))
    return acc_svd

def nearsub_pca(train_features, train_labels, test_features, test_labels, n_comp=10):
    """Perform nearest subspace classification.
    
    Options:
        n_comp (int): number of components for PCA or SVD
    
    """
    scores_pca = []
    classes = np.unique(test_labels)
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
        scores_pca.append(score_pca_j)
    test_predict_pca = np.argmin(scores_pca, axis=0)
    acc_pca = compute_accuracy(classes[test_predict_pca], test_labels)
    print('PCA: {}'.format(acc_pca))
    return acc_svd

def compute_accuracy(y_pred, y_true):
    """Compute accuracy by counting correct classification. """
    assert y_pred.shape == y_true.shape
    return 1 - np.count_nonzero(y_pred - y_true) / y_true.size

def baseline(train_features, train_labels, test_features, test_labels):
    test_models = {'log_l2': SGDClassifier(loss='log', max_iter=10000, random_state=42),
                   'SVM_linear': LinearSVC(max_iter=10000, random_state=42),
                   'SVM_RBF': SVC(kernel='rbf', random_state=42),
                   'DecisionTree': DecisionTreeClassifier(),
                   'RandomForrest': RandomForestClassifier()}
    for model_name in test_models:
        test_model = test_models[model_name]
        test_model.fit(train_features, train_labels)
        score = test_model.score(test_features, test_labels)
        print(f"{model_name}: {score}")
