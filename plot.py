import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.nn.functional import normalize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.decomposition import TruncatedSVD, PCA
import functionals as F
import utils


def plot_heatmap(features, labels, title, model_dir):
    """Plot heatmap of cosine simliarity for all features. """
    num_samples = features.shape[0]
    classes = np.arange(np.unique(labels).size)
    features_sort_, _ = utils.sort_dataset(features, labels, 
                            classes=classes, stack=True)
    sim_mat = np.abs(features_sort_ @ features_sort_.T)
    print(sim_mat.min(), sim_mat.max())

#    plt.rc('text', usetex=True)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    fig, ax = plt.subplots(figsize=(8, 7), sharey=True, sharex=True)
    im = ax.imshow(sim_mat, cmap='Blues')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = fig.colorbar(im, cax=cax, drawedges=0, ticks=[0, 0.5, 1])
    cbar.ax.tick_params(labelsize=18)
    # fig.colorbar(im, pad=0.02, drawedges=0, ticks=[0, 0.5, 1])
    ax.set_xticks(np.linspace(0, num_samples, len(classes)+1))
    ax.set_yticks(np.linspace(0, num_samples, len(classes)+1))
    [tick.label.set_fontsize(24) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(24) for tick in ax.yaxis.get_major_ticks()]
    fig.tight_layout()
    
    save_dir = os.path.join(model_dir, "figures", "heatmaps")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f"heatmap-{title}.pdf"))
    plt.close()

def plot_combined_loss(model_dir, update=None):
    """Plot theoretical loss and empirical loss. 

    Figure 3: gaussian2d, gaussian3d, fontsize 24

    """
#    plt.rc('text', usetex=True)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    fig, ax = plt.subplots(1, 1, figsize=(7, 5), sharey=True, sharex=True)
    models = ['train', 'test']
    linestyles = ['solid', 'dashed']
    markers = ['o', 'D']
    markersizes = [4.5, 3]
    alphas = [0.5, 0.9]
    names = ['$\Delta R$ ', '$R$', '$R_c$']
    colors = ['green', 'royalblue', 'coral']
    for model, linestyle, marker, alpha, markersize in zip(models, linestyles, markers, alphas, markersizes):
        filename = os.path.join(model_dir, "loss", f'{model}.csv')
        data = pd.read_csv(filename)
        losses = [data['loss_total'].ravel(), data['loss_expd'].ravel(), data['loss_comp'].ravel()]
        for loss, name, color in zip(losses, names, colors):
            num_iter = np.arange(loss.size)
            ax.plot(num_iter, loss, label=r'{} ({})'.format(name, model), 
                color=color, linewidth=1.5, alpha=alpha, linestyle=linestyle,
                marker=marker, markersize=markersize, markevery=5, markeredgecolor='black')
    ax.set_ylabel('Loss', fontsize=40)
    ax.set_xlabel('Layers', fontsize=40)
    # ax.set_ylim((-0.05, 2.8)) # gaussian2d
    # ax.set_yticks(np.linspace(0, 2.5, 6)) # gaussian2d
    # ax.set_ylim((-0.05, 2.5)) # gaussian2d
    # ax.set_yticks(np.linspace(0, 2.5, 6)) # gaussian2d
    # ax.set_ylim((0, 4.0)) # gaussian3d
    # ax.set_yticks(np.linspace(0, 4.0, 9)) # gaussian3d
    # ax.set_ylim((-0.005, 0.075)) # mnist_rotation_classes01
    # ax.set_yticks(np.linspace(0, 0.075, 6)) # mnist_rotation_classes01
    # ax.set_ylim((-0.02, 0.1)) # sinusoid
    # ax.set_yticks(np.linspace(0, 0.1, 5)) # sinusoid
    handles, labels = ax.get_legend_handles_labels()
    handles = [handles[i] for i in [0, 3, 1, 4, 2, 5]]
    labels = [labels[i] for i in [0, 3, 1, 4, 2, 5]]
    ax.legend(handles, labels, loc='lower right', prop={"size": 13}, ncol=3, framealpha=0.5)
    [tick.label.set_fontsize(22) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(22) for tick in ax.yaxis.get_major_ticks()]
    fig.tight_layout()
    
    save_dir = os.path.join(model_dir, 'figures', 'loss')
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.join(save_dir, f'loss-traintest.pdf')
    plt.savefig(file_name, dpi=200)
    plt.close()

def plot_2d(Z, y, name, model_dir):
    plot_dir = os.path.join(model_dir, "figures", "2dscatter")
    colors = np.array(['forestgreen', 'red', 'royalblue', 'purple', 'darkblue', 'orange'])
    os.makedirs(plot_dir, exist_ok=True)
#    plt.rc('text', usetex=True)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    # colors = np.array(['royalblue', 'forestgreen', 'red'])
    fig, ax = plt.subplots(figsize=(6, 5), dpi=200)
    ax.scatter(Z[:, 0], Z[:, 1], c=colors[y], alpha=0.5)
    ax.scatter(0.0, 0.0, c='black', alpha=0.8, marker='s')
    # ax.arrow(0.0, 0.0, Z[:, 0], Z[:, 1])
    ax.set_ylim(-1.2, 1.2)
    ax.set_xlim(-1.2, 1.2)
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.grid(linestyle=':')
    Z, _ = F.get_n_each(Z, y, 1)
    for c in np.unique(y):
        ax.arrow(0, 0, Z[c, 0], Z[c, 1], head_width=0.03, head_length=0.05, fc='k', ec='k', length_includes_head=True)
    [tick.label.set_fontsize(24) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(24) for tick in ax.yaxis.get_major_ticks()]
    plt.savefig(os.path.join(plot_dir, "scatter2d-"+name+".pdf"), dpi=200)
    plt.close()

def plot_3d(Z, y, name, model_dir):
    colors = np.array(['green', 'blue', 'red'])
    savedir = os.path.join(model_dir, 'figures', '3d')
    os.makedirs(savedir, exist_ok=True)
#    plt.rc('text', usetex=True)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    colors = np.array(['forestgreen', 'royalblue', 'brown'])
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(Z[:, 0], Z[:, 1], Z[:, 2], c=colors[y], cmap=plt.cm.Spectral, s=200.0)
    Z, _ = F.get_n_each(Z, y, 1)
    for c in np.unique(y):
        ax.quiver(0.0, 0.0, 0.0, Z[c, 0], Z[c, 1], Z[c, 2], length=1.0, normalize=True, arrow_length_ratio=0.05, color='black')
    u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
    x = np.cos(u)*np.sin(v)
    y = np.sin(u)*np.sin(v)
    z = np.cos(v)
    ax.plot_wireframe(x, y, z, color="gray", alpha=0.5)
    ax.xaxis._axinfo["grid"]['color'] =  (0,0,0,0.1)
    ax.yaxis._axinfo["grid"]['color'] =  (0,0,0,0.1)
    ax.zaxis._axinfo["grid"]['color'] =  (0,0,0,0.1)
    [tick.label.set_fontsize(24) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(24) for tick in ax.yaxis.get_major_ticks()]
    [tick.label.set_fontsize(24) for tick in ax.zaxis.get_major_ticks()]
    ax.view_init(20, 15)
    plt.tight_layout()
    fig.savefig(os.path.join(savedir, f"scatter3d-{name}.jpg"), dpi=200)
    plt.close()

def plot_sample_angle_combined(train_features, train_labels, test_features, test_labels, model_dir, title1, title2, tail=""):
    save_dir = os.path.join(model_dir, "figures", "sample_angle_combined")
    os.makedirs(save_dir, exist_ok=True)
    
    colors = ['blue', 'red', 'green']
    _bins = np.linspace(-0.05, 1.05, 21)

    classes = np.unique(y_train)
    fs_train, _ = utils.sort_dataset(train_features, train_labels, 
                        classes=classes, stack=False)
    fs_test, _ = utils.sort_dataset(test_features, test_labels, 
                            classes=classes, stack=False)
    angles = []
    for class_train in classes:
        for class_test in classes:
            if class_train == class_test:
                continue
            angles.append((fs_train[class_train] @ fs_test[class_test].T).reshape(-1))

#    plt.rc('text', usetex=True)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(np.hstack(angles), bins=_bins, alpha=0.5,   color='red', #colors[class_test], 
                edgecolor='black')#, label=f'Class {class_test}')
    ax.set_xlabel('Similarity', fontsize=38)
    ax.set_ylabel('Count', fontsize=38)
    ax.ticklabel_format(style='sci', scilimits=(0, 3))
    [tick.label.set_fontsize(22) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(22) for tick in ax.yaxis.get_major_ticks()]
    # ax.legend(loc='upper center', prop={"size": 13}, ncol=1, framealpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'sample_angle_combined-{title1}-vs-{title2}{tail}.pdf'))
    plt.close()