import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn.functional import normalize
from mpl_toolkits.mplot3d import Axes3D
from MulticoreTSNE import MulticoreTSNE as TSNE
from sklearn.decomposition import TruncatedSVD
import train_func as tf
import utils


def plot_heatmap(features, labels, title, classes, model_dir):
    """Plot heatmap of cosine simliarity for all features. """
    num_samples = features.shape[0]
    if type(classes) == int:
        classes = np.arange(classes)
    features_sort_, _ = utils.sort_dataset(features, labels, 
                            classes=classes, stack=True)
    print(features_sort_.shape)
    # features_sort_ = np.vstack(features_sort)
    sim_mat = np.abs(features_sort_ @ features_sort_.T)
    print(sim_mat.min(), sim_mat.max())

    fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    im = ax.imshow(sim_mat, cmap='Blues')
    fig.colorbar(im, pad=0.02, drawedges=0, ticks=[0, 0.5, 1])
    ax.set_xticks(np.linspace(0, num_samples, len(classes)))
    ax.set_yticks(np.linspace(0, num_samples, len(classes)))
    [tick.label.set_fontsize(12) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(12) for tick in ax.yaxis.get_major_ticks()]
    fig.tight_layout()
    
    save_dir = os.path.join(model_dir, "figures", "heatmaps")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"heatmap-{title}.png"), dpi=200)
    plt.close()

def plot_combined_loss(model_dir):
    """Plot theoretical loss and empirical loss. """
    ## extract loss from csv
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True, sharex=True)
    models = ['train', 'test']
    linestyles = ['solid', 'dashed']
    for model, linestyle in zip(models, linestyles):
        filename = os.path.join(model_dir, "loss", f'{model}.csv')
        data = pd.read_csv(filename)
        expd = data['loss_expd'].ravel()
        comp = data['loss_comp'].ravel()
        total = data['loss_total'].ravel()

        num_iter = np.arange(total.size)
        ax.plot(num_iter, total, label=r'$\Delta R$ ({})'.format(model), 
            color='green', linewidth=1.0, alpha=0.8, linestyle=linestyle)
        ax.plot(num_iter, expd, label=r'$R$  ({})'.format(model), 
                color='royalblue', linewidth=1.0, alpha=0.8, linestyle=linestyle)
        ax.plot(num_iter, comp, label=r'$R^c$  ({})'.format(model), 
                color='coral', linewidth=1.0, alpha=0.8, linestyle=linestyle)

    ax.set_ylabel('Loss', fontsize=10)
    ax.set_xlabel('Number of iterations', fontsize=10)
    ax.legend(loc='best', prop={"size": 5}, ncol=3, framealpha=0.5)
    plt.tight_layout()
    
    save_dir = os.path.join(model_dir, 'figures', 'loss')
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, f'loss-traintest.png')
    plt.savefig(file_name, dpi=200)
    plt.close()

def plot_2d(Z, y, name, model_dir):
    plot_dir = os.path.join(model_dir, "figures", "2dscatter")
    colors = np.array(['green', 'red', 'blue'])
    os.makedirs(plot_dir, exist_ok=True)
    plt.scatter(Z[:, 0], Z[:, 1], c=colors[y], alpha=0.8)
    # plt.title(name)
    plt.ylim(-1.2, 1.2)
    plt.xlim(-1.2, 1.2)
    plt.xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    plt.yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    plt.savefig(os.path.join(plot_dir, "scatter2d-"+name+".png"), dpi=200)
    plt.close()

def plot_3d(X, y, name, model_dir):
    colors = np.array(['green', 'blue', 'red'])
    savedir = os.path.join(model_dir, 'figures', '3d')
    os.makedirs(savedir, exist_ok=True)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors[y], cmap=plt.cm.Spectral)
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, f"scatter3d-{name}.png"), dpi=200)
    plt.close()

def plot_nearsub_angle(X_train, y_train, Z_train, X_test, y_test, Z_test, n_comp, model_dir):
    save_dir = os.path.join(model_dir, "figures", "subspace_angle")
    os.makedirs(save_dir, exist_ok=True)
    
    Z_test = tf.normalize(Z_test.reshape(Z_test.shape[0], -1))
    Z_train = tf.normalize(Z_train.reshape(Z_train.shape[0], -1))
    X_test = tf.normalize(X_test.reshape(X_test.shape[0], -1))
    X_train = tf.normalize(X_train.reshape(X_train.shape[0], -1))
    print(X_train.shape, Z_train.shape)
    
    # with X
    fd = X_train.shape[1]
    if n_comp >= fd:
        n_comp = fd - 1
    classes = np.unique(y_train)
    for subspace_class in classes:
        scores = []
        for c in classes:
            X_train_c = X_train[y_train==subspace_class]
            X_test_c = X_test[y_test==c]
            svd = TruncatedSVD(n_components=n_comp).fit(X_train_c)
            svd_subspace = svd.components_.T
            svd_j = (np.eye(fd) - svd_subspace @ svd_subspace.T) \
                                @ (X_test_c).T
            score_svd_j = np.linalg.norm(svd_j, ord=2, axis=0)
            scores.append(score_svd_j)

        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
        ax.hist(scores[0], bins=np.linspace(0, 1, 50), alpha=0.8, color='blue')
        ax.hist(scores[1], bins=np.linspace(0, 1, 50), alpha=0.8, color='red')
        ax.set_xlabel('similarity')
        ax.set_ylabel('count')
        ax.set_xlim([0, 1])
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'subspace_angle-X-subspace{subspace_class}'))
        plt.close()

    # with Z
    fd = Z_train.shape[1]
    if n_comp >= fd:
        n_comp = fd - 1
    classes = np.unique(y_train)
    for subspace_class in classes:
        scores = []
        for c in classes:
            Z_train_c = Z_train[y_train==subspace_class]
            Z_test_c = Z_test[y_test==c]
            svd = TruncatedSVD(n_components=n_comp).fit(Z_train_c)
            svd_subspace = svd.components_.T
            svd_j = (np.eye(fd) - svd_subspace @ svd_subspace.T) \
                                @ (Z_test_c).T
            score_svd_j = np.linalg.norm(svd_j, ord=2, axis=0)
            scores.append(score_svd_j)
    
        fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
        ax.hist(scores[0], bins=np.linspace(0, 1, 100), alpha=0.8, color='blue')
        ax.hist(scores[1], bins=np.linspace(0, 1, 100), alpha=0.8, color='red')
        ax.set_xlabel('similarity')
        ax.set_ylabel('count')
        ax.set_xlim([0, 1])
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'subspace_angle-Z-subspace{subspace_class}'))
        plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=int, help='model directory')
    parser.add_argument('--classes', type=str, help='classes')

    parser.add_argument('--loss', action='store_true', help='plot loss from training and testing')
    parser.add_argument('--heatmap', action='store_true', help='plot heatmaps')
    parser.add_argument('--angle', action='store_true', help='plot angle between subspace and samples')
    args = parser.parse_args()

    # load features
    X_train = np.load(os.path.join(model_dir, "features", "X_train_features.npy"))
    y_train = np.load(os.path.join(model_dir, "features", "Z_train_labels.npy"))
    Z_train = np.load(os.path.join(model_dir, "features", "Z_train_features.npy"))
    
    X_test = np.load(os.path.join(model_dir, "features", "X_test_features.npy"))
    y_test = np.load(os.path.join(model_dir, "features", "Z_test_labels.npy"))
    Z_test = np.load(os.path.join(model_dir, "features", "Z_test_features.npy"))

    classes = np.array(list(args.classes)).astype(np.int32)
    X_train = tf.normalize(X_train.reshape(X_train.shape[0], -1))
    X_test = tf.normalize(X_test.reshape(X_test.shape[0], -1))
    Z_train = tf.normalize(Z_train.reshape(Z_train.shape[0], -1))
    Z_test = tf.normalize(Z_test.reshape(Z_test.shape[0], -1))

    # plot
    if args.loss:
        plot_combined_loss(model_dir)
    if args.heatmap:
        plot_heatmap(X_train, y_train, "X_train", classes, model_dir)
        plot_heatmap(X_test, y_test, "X_test", classes, model_dir)
        plot_heatmap(Z_train, y_train, "Z_train", classes, model_dir)
        plot_heatmap(Z_test, y_test, "Z_test", classes, model_dir)
        plot_heatmap(Z_translate, y_translate, "Z_translate", classes, model_dir)
    

    if len(X_train.shape) > 2: # multichannel data
        X_translate = np.load(os.path.join(model_dir, "features", "X_translate_features.npy"))
        y_translate = np.load(os.path.join(model_dir, "features", "Z_translate_labels.npy"))
        Z_translate = np.load(os.path.join(model_dir, "features", "Z_translate_features.npy"))
        X_translate = tf.normalize(X_translate.reshape(X_translate.shape[0], -1))
        Z_translate = tf.normalize(Z_translate.reshape(Z_translate.shape[0], -1))

        if args.heatmap:
            plot_heatmap(X_translate, y_translate, "X_translate", classes, model_dir)
            plot_nearsub_angle(X_train, y_train, Z_train, X_translate, y_translate, Z_translate, 20, model_dir)