import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn.functional import normalize
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import TruncatedSVD, PCA
import train_func as tf
import utils


def plot_heatmap(features, labels, title, classes, model_dir):
    """Plot heatmap of cosine simliarity for all features. """
    num_samples = features.shape[0]
    classes = np.arange(np.unique(labels).size)
    features_sort_, _ = utils.sort_dataset(features, labels, 
                            classes=classes, stack=True)
    sim_mat = np.abs(features_sort_ @ features_sort_.T)
    print(sim_mat.min(), sim_mat.max())

    plt.rc('text', usetex=True)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    im = ax.imshow(sim_mat, cmap='Blues')
    fig.colorbar(im, pad=0.02, drawedges=0, ticks=[0, 0.5, 1])
    ax.set_xticks(np.linspace(0, num_samples, len(classes)+1))
    ax.set_yticks(np.linspace(0, num_samples, len(classes)+1))
    [tick.label.set_fontsize(14) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(14) for tick in ax.yaxis.get_major_ticks()]
    fig.tight_layout()
    
    save_dir = os.path.join(model_dir, "figures", "heatmaps")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"heatmap-{title}.pdf"), dpi=200)
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
    file_name = os.path.join(save_dir, f'loss-traintest.pdf')
    plt.savefig(file_name, dpi=200)
    plt.close()

def plot_2d(Z, y, name, model_dir):
    plot_dir = os.path.join(model_dir, "figures", "2dscatter")
    colors = np.array(['green', 'red', 'blue'])
    os.makedirs(plot_dir, exist_ok=True)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    colors = np.array(['royalblue', 'forestgreen', 'blue'])
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    ax.scatter(Z[:, 0], Z[:, 1], c=colors[y], alpha=0.5)
    ax.scatter(0.0, 0.0, c='black', alpha=0.8, marker='s')
    # ax.arrow(0.0, 0.0, Z[:, 0], Z[:, 1])
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlim(-1.2, 1.2)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.grid(linestyle=':')
    Z, _ = tf.get_n_each(Z, y, 1)
    ax.arrow(0, 0, Z[0, 0], Z[0, 1], head_width=0.03, head_length=0.05, fc='k', ec='k', length_includes_head=True)
    ax.arrow(0, 0, Z[-1, 0], Z[-1, 1], head_width=0.03, head_length=0.05, fc='k', ec='k', length_includes_head=True)
    [tick.label.set_fontsize(14) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(14) for tick in ax.yaxis.get_major_ticks()]
    plt.savefig(os.path.join(plot_dir, "scatter2d-"+name+".pdf"), dpi=200)
    plt.close()

def plot_3d(Z, y, name, model_dir):
    colors = np.array(['green', 'blue', 'red'])
    savedir = os.path.join(model_dir, 'figures', '3d')
    os.makedirs(savedir, exist_ok=True)
    plt.rc('text', usetex=True)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    colors = np.array(['forestgreen', 'royalblue', 'brown'])
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=colors[y], cmap=plt.cm.Spectral, s=200.0)
    Z, _ = tf.get_n_each(Z, y, 1)
    ax.scatter(0.0, 0.0, 0.0, c='black', cmap=plt.cm.Spectral, marker='s', s=50.0)
    ax.quiver(0.0, 0.0, 0.0, Z[0, 0], Z[0, 1], Z[0, 2], length=1.0, normalize=True, arrow_length_ratio=0.05, color='black')
    ax.quiver(0.0, 0.0, 0.0, Z[1, 0], Z[1, 1], Z[1, 2], length=1.0, normalize=True, arrow_length_ratio=0.05, color='black')
    ax.quiver(0.0, 0.0, 0.0, Z[2, 0], Z[2, 1], Z[2, 2], length=1.0, normalize=True, arrow_length_ratio=0.05, color='black')
    [tick.label.set_fontsize(14) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(14) for tick in ax.yaxis.get_major_ticks()]
    [tick.label.set_fontsize(14) for tick in ax.zaxis.get_major_ticks()]
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, f"scatter3d-{name}.pdf"), dpi=200)
    plt.close()

def plot_nearsub_angle(X_train, y_train, Z_train, X_test, y_test, Z_test, n_comp, model_dir):
    save_dir = os.path.join(model_dir, "figures", "subspace_angle")
    os.makedirs(save_dir, exist_ok=True)
    
    Z_test = tf.normalize(Z_test.reshape(Z_test.shape[0], -1))
    Z_train = tf.normalize(Z_train.reshape(Z_train.shape[0], -1))
    X_test = tf.normalize(X_test.reshape(X_test.shape[0], -1))
    X_train = tf.normalize(X_train.reshape(X_train.shape[0], -1))
    
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
        fig.savefig(os.path.join(save_dir, f'subspace_angle-dim{n_comp}-X-subspace{subspace_class}.pdf'))
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
        fig.savefig(os.path.join(save_dir, f'subspace_angle-dim{n_comp}-Z-subspace{subspace_class}.pdf'))
        plt.close()

def plot_pca(features, labels, n_comp, title, classes, model_dir):
    pca_dir = os.path.join(model_dir, 'figures', 'pca')
    os.makedirs(pca_dir, exist_ok=True)

    if type(classes) == int:
        classes = np.arange(classes)
    features_sort, _ = utils.sort_dataset(features, labels, 
                            classes=classes, stack=False)

    pca = PCA(n_components=n_comp).fit(features)
    sig_vals = [pca.singular_values_]
    for c in classes: 
        pca = PCA(n_components=n_comp).fit(features_sort[c])
        sig_vals.append((pca.singular_values_))

    ## plot features
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(7, 5), dpi=500)
    x_min = np.min([len(sig_val) for sig_val in sig_vals])
    # ax.plot(np.arange(x_min), sig_vals[0][:x_min], '-p', markersize=3, markeredgecolor='black',
    #     linewidth=1.5, color='tomato')
    map_vir = plt.cm.get_cmap('Blues', 6)
    norm = plt.Normalize(-10, 10)
    norm_class = norm(classes)
    color = map_vir(norm_class)
    for c, sig_val in enumerate(sig_vals[1:]):
        ax.plot(np.arange(x_min), sig_val[:x_min], '-o', markersize=3, markeredgecolor='black',
                alpha=0.6, linewidth=1.0, color=color[c])
    ax.set_xticks(np.arange(0, x_min, 5))
    ax.set_yticks(np.arange(0, 35, 5))
    ax.set_xlabel("components", fontsize=14)
    ax.set_ylabel("sigular values", fontsize=14)
    [tick.label.set_fontsize(12) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(12) for tick in ax.yaxis.get_major_ticks()]
    fig.tight_layout()
    
    file_name = os.path.join(pca_dir, f"pca_{title}.pdf")
    fig.savefig(file_name)
    plt.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, help='model directory')
    parser.add_argument('--classes', type=str, help='classes')
    parser.add_argument('--n_comp', type=int, default=50, help='number of components')

    parser.add_argument('--loss', action='store_true', help='plot loss from training and testing')
    parser.add_argument('--scatter', action='store_true', help='plot scatter')
    parser.add_argument('--heatmap', action='store_true', help='plot heatmaps')
    parser.add_argument('--angle', action='store_true', help='plot angle between subspace and samples')
    parser.add_argument('--pca', action='store_true', help='plot pca')
    args = parser.parse_args()

    params = utils.load_params(args.model_dir)
    X_train = np.load(os.path.join(args.model_dir, "features", "X_train_features.npy"))
    y_train = np.load(os.path.join(args.model_dir, "features", "Z_train_labels.npy"))
    Z_train = np.load(os.path.join(args.model_dir, "features", "Z_train_features.npy"))
    
    X_test = np.load(os.path.join(args.model_dir, "features", "X_test_features.npy"))
    y_test = np.load(os.path.join(args.model_dir, "features", "Z_test_labels.npy"))
    Z_test = np.load(os.path.join(args.model_dir, "features", "Z_test_features.npy"))

    classes = np.unique(y_train)
    multichannel = len(X_train.shape) > 2
    X_train = tf.normalize(X_train.reshape(X_train.shape[0], -1))
    X_test = tf.normalize(X_test.reshape(X_test.shape[0], -1))
    Z_train = tf.normalize(Z_train.reshape(Z_train.shape[0], -1))
    Z_test = tf.normalize(Z_test.reshape(Z_test.shape[0], -1))

    # plot
    if args.loss:
        plot_combined_loss(args.model_dir)
    if args.heatmap:
        plot_heatmap(X_train, y_train, "X_train", classes, args.model_dir)
        plot_heatmap(X_test, y_test, "X_test", classes, args.model_dir)
        plot_heatmap(Z_train, y_train, "Z_train", classes, args.model_dir)
        plot_heatmap(Z_test, y_test, "Z_test", classes, args.model_dir)
    if args.pca:
        plot_pca(X_train, y_train, args.n_comp, "X", classes, args.model_dir)
        plot_pca(Z_train, y_train, args.n_comp, "Z", classes, args.model_dir)
    if args.scatter:
        if X_train.shape[1] == 2:
            plot_2d(X_train, y_train, "X_train", args.model_dir)
            plot_2d(X_test, y_test, "X_test", args.model_dir)
            plot_2d(Z_train, y_train, "Z_train", args.model_dir)
            plot_2d(Z_test, y_test, "Z_test", args.model_dir)
        elif X_train.shape[1] == 3:
            plot_3d(X_train, y_train, "X_train", args.model_dir)
            plot_3d(X_test, y_test, "X_test", args.model_dir)
            plot_3d(Z_train, y_train, "Z_train", args.model_dir)
            plot_3d(Z_test, y_test, "Z_test", args.model_dir)

    if multichannel: # multichannel data
        X_translate = np.load(os.path.join(args.model_dir, "features", "X_translate_features.npy"))
        y_translate = np.load(os.path.join(args.model_dir, "features", "X_translate_labels.npy"))
        Z_translate = np.load(os.path.join(args.model_dir, "features", "Z_translate_features.npy"))
        X_translate = tf.normalize(X_translate.reshape(X_translate.shape[0], -1))
        Z_translate = tf.normalize(Z_translate.reshape(Z_translate.shape[0], -1))

        if args.heatmap:
            plot_heatmap(X_translate, y_translate, "X_translate", classes, args.model_dir)
            plot_heatmap(Z_translate, y_translate, "Z_translate", classes, args.model_dir)
        if args.angle:
            plot_nearsub_angle(X_train, y_train, Z_train, 
                               X_translate, y_translate, Z_translate, 
                               args.n_comp, args.model_dir)
#            plot_nearsub_angle(X_train, y_train, Z_train, X_test, y_test, Z_test, args.n_comp, args.model_dir)
