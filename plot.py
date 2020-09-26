import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn.functional import normalize
from mpl_toolkits.mplot3d import Axes3D
from MulticoreTSNE import MulticoreTSNE as TSNE
import train_func as tf
import utils


def plot_heatmap(features, labels, title, num_classes, model_dir):
    """Plot heatmap of cosine simliarity for all features. """
    num_samples = features.shape[0]
    features_sort, _ = utils.sort_dataset(features, labels, 
                            num_classes=num_classes, stack=False)
    features_sort_ = np.vstack(features_sort)
    sim_mat = np.abs(features_sort_ @ features_sort_.T)
    print(sim_mat.min(), sim_mat.max())

    fig, ax = plt.subplots(figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    im = ax.imshow(sim_mat, cmap='Blues')
    fig.colorbar(im, pad=0.02, drawedges=0, ticks=[0, 0.5, 1])
    ax.set_xticks(np.linspace(0, num_samples, 6))
    ax.set_yticks(np.linspace(0, num_samples, 6))
    [tick.label.set_fontsize(10) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(10) for tick in ax.yaxis.get_major_ticks()]
    fig.tight_layout()
    
    save_dir = os.path.join(model_dir, "figures", "heatmaps")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"heatmap-{title}.png"), dpi=200)
    plt.close()


def plot_combined_loss(model_dir):
    """Plot theoretical loss and empirical loss. """
    ## extract loss from csv
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True, sharex=True, dpi=400)
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


def plot_combined_loss_translation(model_dir):
    """Plot theoretical loss and empirical loss. """
    ## extract loss from csv
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True, sharex=True, dpi=400)
    models = ["init", "train", "translate"]
    linestyles = ['dotted', 'solid', 'dashed']
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
    file_name = os.path.join(save_dir, f'loss-translate.png')
    plt.savefig(file_name, dpi=200)
    plt.close()


def plot_tsne(features, labels, title, save_dir):
    save_dir = os.path.join(save_dir, "tsne")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, f"plot-tsne-{title}.png")

    num_classes = np.unique(labels).size
    embeddings = TSNE(n_jobs=4).fit_transform(features)
    vis_x = embeddings[:, 0]
    vis_y = embeddings[:, 1]
    plt.scatter(vis_x, vis_y, c=labels, cmap=plt.cm.get_cmap("jet", num_classes), marker='.')
    plt.colorbar(ticks=range(num_classes))
    plt.clim(-0.5, 9.5)
    plt.savefig(save_path, dpi=200)
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
    plt.savefig(os.path.join(plot_dir, "scatter2d-"+name+".png"))
    plt.close()

def plot_3d(X, y, name, model_dir):
    colors = np.array(['green', 'blue', 'red'])
    savedir = os.path.join(model_dir, 'figures', '3d')
    os.makedirs(savedir, exist_ok=True)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=colors[y], cmap=plt.cm.Spectral)
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, f"scatter3d-{name}.png"))
    plt.close()
