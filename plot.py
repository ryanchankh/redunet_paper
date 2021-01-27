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
    fig, ax = plt.subplots(figsize=(8, 7), sharey=True, sharex=True, dpi=400)
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
    plt.savefig(os.path.join(save_dir, f"heatmap-{title}.pdf"), dpi=200)
    plt.close()

def plot_combined_loss(model_dir):
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
                marker=marker, markersize=markersize, markevery=100, markeredgecolor='black')
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
    plt.rc('text', usetex=True)
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
    plt.rc('text', usetex=True)
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

def plot_3d_transformations(model_dir):
    features_dir = os.path.join(model_dir, 'features', 'layers')
    for filename in os.listdir(features_dir):
        if filename[-4:] != '.npy':
            continue
        Z_layer = np.load(os.path.join(features_dir, filename))
        y_layer = np.load(os.path.join(model_dir, 'features', 'X_train_labels.npy'))
        plot_3d(Z_layer, y_layer, filename[:-4], model_dir)
        print(filename)


def plot_nearsub_angle(train_features, train_labels, test_features, test_labels, 
                       n_comp, model_dir, title, tail=""):
    def least_square(train_features, test_features, n_comp):
        U, S, V = np.linalg.svd(train_features)
        U = U[:, :n_comp]
        S = np.diag(S[:n_comp])
        V = V[:n_comp, :] 
        X = U @ S @ V

        theta, r, rank, sig_val = np.linalg.lstsq(X.T, test_features.T, rcond=-1)
        out = theta.T @ X - test_features
        residual = np.linalg.norm(out, ord=2, axis=1)
        return residual
    save_dir = os.path.join(model_dir, "figures", "subspace_angle")
    os.makedirs(save_dir, exist_ok=True)
    
    colors = ['blue', 'red', 'green']
    classes = np.unique(y_train)
    fs_train, _ = utils.sort_dataset(train_features, train_labels, 
                        classes=classes, stack=False)
    fs_test, _ = utils.sort_dataset(test_features, test_labels, 
                            classes=classes, stack=False)
    for class_train in classes:
#        plt.rc('text', usetex=True)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        fig, ax = plt.subplots(figsize=(7, 5))
        
        for class_test in classes:
            _bins = np.linspace(-0.05, 1.05, 21)
            residuals = least_square(fs_train[class_train], fs_test[class_test], n_comp)
            print('class_train', class_train, 'class_test', class_test, 'residuals', residuals[:10])
            ax.hist(residuals, bins=_bins, alpha=0.5, color=colors[class_test], 
                    edgecolor='black', label=f'Class {class_test}')

        ax.set_xlabel('Distance', fontsize=22)
        ax.set_ylabel('Count', fontsize=22)
        [tick.label.set_fontsize(22) for tick in ax.xaxis.get_major_ticks()] 
        [tick.label.set_fontsize(22) for tick in ax.yaxis.get_major_ticks()]
        ax.legend(loc='upper center', prop={"size": 13}, ncol=1, framealpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'subspace_angle-dim{n_comp}-{title}-subspace{class_train}{tail}.pdf'), dpi=200)
        plt.close()

def plot_nearsub_angle(train_features, train_labels, test_features, test_labels, 
                       n_comp, model_dir, title, tail=""):
    def least_square(train_features, test_features, n_comp):
        U, S, V = np.linalg.svd(train_features)
        U = U[:, :n_comp]
        S = np.diag(S[:n_comp])
        V = V[:n_comp, :] 
        X = U @ S @ V

        theta, r, rank, sig_val = np.linalg.lstsq(X.T, test_features.T, rcond=-1)
        out = theta.T @ X - test_features
        residual = np.linalg.norm(out, ord=2, axis=1)
        return residual
    save_dir = os.path.join(model_dir, "figures", "subspace_angle")
    os.makedirs(save_dir, exist_ok=True)
    
    colors = ['blue', 'red', 'green']
    classes = np.unique(y_train)
    fs_train, _ = utils.sort_dataset(train_features, train_labels, 
                        classes=classes, stack=False)
    fs_test, _ = utils.sort_dataset(test_features, test_labels, 
                            classes=classes, stack=False)
    for class_train in classes:
        plt.rc('text', usetex=True)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        fig, ax = plt.subplots(figsize=(7, 5))
        
        for class_test in classes:
            _bins = np.linspace(-0.05, 1.05, 21)
            residuals = least_square(fs_train[class_train], fs_test[class_test], n_comp)
            print('class_train', class_train, 'class_test', class_test, 'residuals', residuals[:10])
            ax.hist(residuals, bins=_bins, alpha=0.5, color=colors[class_test], 
                    edgecolor='black', label=f'Class {class_test}')

        ax.set_xlabel('Distance', fontsize=22)
        ax.set_ylabel('Count', fontsize=22)
        [tick.label.set_fontsize(22) for tick in ax.xaxis.get_major_ticks()] 
        [tick.label.set_fontsize(22) for tick in ax.yaxis.get_major_ticks()]
        ax.legend(loc='upper center', prop={"size": 13}, ncol=1, framealpha=0.5)
        fig.tight_layout()
        fig.savefig(os.path.join(save_dir, f'subspace_angle-dim{n_comp}-{title}-subspace{class_train}{tail}.pdf'), dpi=200)
        plt.close()

def plot_sample_angle(train_features, train_labels, test_features, test_labels, model_dir, title1, title2, tail=""):
    save_dir = os.path.join(model_dir, "figures", "sample_angle")
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
        plt.rc('text', usetex=True)
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman']
        fig, ax = plt.subplots(figsize=(7, 5))
        for class_test in classes:
            if class_train == class_test:
                continue
            angles = (fs_train[class_train] @ fs_test[class_test].T).reshape(-1)
            ax.hist(np.hstack(angles), bins=_bins, alpha=0.5, color=colors[class_test], 
                        edgecolor='black')#, label=f'Class {class_test}')
            ax.set_xlabel('Similarity', fontsize=38)
            ax.set_ylabel('Count', fontsize=38)
            ax.ticklabel_format(style='sci', scilimits=(0, 3))
            [tick.label.set_fontsize(22) for tick in ax.xaxis.get_major_ticks()] 
            [tick.label.set_fontsize(22) for tick in ax.yaxis.get_major_ticks()]
            # ax.legend(loc='upper center', prop={"size": 13}, ncol=1, framealpha=0.5)
            fig.tight_layout()
            fig.savefig(os.path.join(save_dir, f'sample_angle-{title1}{class_train}-vs-{title2}{class_test}{tail}.pdf'), dpi=200)
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

    plt.rc('text', usetex=True)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(np.hstack(angles), bins=_bins, alpha=0.5, color=colors[class_test], 
                edgecolor='black')#, label=f'Class {class_test}')
    ax.set_xlabel('Similarity', fontsize=38)
    ax.set_ylabel('Count', fontsize=38)
    ax.ticklabel_format(style='sci', scilimits=(0, 3))
    [tick.label.set_fontsize(22) for tick in ax.xaxis.get_major_ticks()] 
    [tick.label.set_fontsize(22) for tick in ax.yaxis.get_major_ticks()]
    # ax.legend(loc='upper center', prop={"size": 13}, ncol=1, framealpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(save_dir, f'sample_angle_combined-{title1}-vs-{title2}{tail}.pdf'), dpi=200)
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
    parser.add_argument('--tail', type=str, default='', help='ending message')

    parser.add_argument('--loss', action='store_true', help='plot loss from training and testing')
    parser.add_argument('--scatter', action='store_true', help='plot scatter')
    parser.add_argument('--heatmap', action='store_true', help='plot heatmaps')
    parser.add_argument('--subspace_angle', action='store_true', help='plot angle between subspace and samples')
    parser.add_argument('--sample_angle', action='store_true', help='plot angle between train and test samples')
    parser.add_argument('--sample_angle_combined', action='store_true', help='plot angle between train and test samples')
    parser.add_argument('--pca', action='store_true', help='plot pca')
    args = parser.parse_args()

    # params = utils.load_params(args.model_dir)
    X_train = np.load(os.path.join(args.model_dir, "features", "X_train_features.npy"))
    y_train = np.load(os.path.join(args.model_dir, "features", "Z_train_labels.npy"))
    Z_train = np.load(os.path.join(args.model_dir, "features", "Z_train_features.npy"))
    
    X_test = np.load(os.path.join(args.model_dir, "features", "X_test_features.npy"))
    y_test = np.load(os.path.join(args.model_dir, "features", "Z_test_labels.npy"))
    Z_test = np.load(os.path.join(args.model_dir, "features", "Z_test_features.npy"))

    classes = np.unique(y_train)
    multichannel = len(X_train.shape) > 2
    X_train = F.normalize(X_train.reshape(X_train.shape[0], -1))
    X_test = F.normalize(X_test.reshape(X_test.shape[0], -1))
    Z_train = F.normalize(Z_train.reshape(Z_train.shape[0], -1))
    Z_test = F.normalize(Z_test.reshape(Z_test.shape[0], -1))

    # plot
    if args.loss:
        plot_combined_loss(args.model_dir)
    if args.heatmap:
        plot_heatmap(X_train, y_train, "X_train", args.model_dir)
        plot_heatmap(X_test, y_test, "X_test", args.model_dir)
        plot_heatmap(Z_train, y_train, "Z_train", args.model_dir)
        plot_heatmap(Z_test, y_test, "Z_test", args.model_dir)
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
            plot_3d_transformations(args.model_dir)

    if multichannel: # multichannel data
        X_translate = np.load(os.path.join(args.model_dir, "features", "X_translate_features.npy"))
        y_translate = np.load(os.path.join(args.model_dir, "features", "X_translate_labels.npy"))
        Z_translate = np.load(os.path.join(args.model_dir, "features", "Z_translate_features.npy"))
        # X_translate = np.vstack([np.load(os.path.join(args.model_dir, "features", f"X_translate_features-{i}-{i+50}.npy")) for i in range(0, 1000, 50)])
        # y_translate = np.hstack([np.load(os.path.join(args.model_dir, "features", f"y_translate_labels-{i}-{i+50}.npy")) for i in range(0, 1000, 50)])
        # Z_translate = np.vstack([np.load(os.path.join(args.model_dir, "features", f"Z_translate_features-{i}-{i+50}.npy")) for i in range(0, 1000, 50)])
#        X_translate = np.vstack([np.load(os.path.join(args.model_dir, "features", f"X_translate_features-{i}-{i+25}.npy")) for i in range(0, 100, 25)])
#        y_translate = np.hstack([np.load(os.path.join(args.model_dir, "features", f"y_translate_labels-{i}-{i+25}.npy")) for i in range(0, 100, 25)])
#        Z_translate = np.vstack([np.load(os.path.join(args.model_dir, "features", f"Z_translate_features-{i}-{i+25}.npy")) for i in range(0, 100, 25)])
        X_translate = F.normalize(X_translate.reshape(X_translate.shape[0], -1))
        Z_translate = F.normalize(Z_translate.reshape(Z_translate.shape[0], -1))

        if args.heatmap:
            plot_heatmap(X_translate, y_translate, "X_translate", args.model_dir)
            plot_heatmap(Z_translate, y_translate, "Z_translate", args.model_dir)
        if args.subspace_angle:
            plot_nearsub_angle(X_train, y_train, X_test, y_test, 
                               args.n_comp, args.model_dir, "X-test", args.tail)
            plot_nearsub_angle(X_train, y_train, X_translate, y_translate, 
                               args.n_comp, args.model_dir, "X-translate", args.tail)
            plot_nearsub_angle(Z_train, y_train, Z_test, y_test, 
                               args.n_comp, args.model_dir, "Z-test", args.tail)
            plot_nearsub_angle(Z_train, y_train, Z_translate, y_translate, 
                               args.n_comp, args.model_dir, "Z-translate", args.tail)
        if args.sample_angle:
            plot_sample_angle(X_train, y_train, X_test, y_test, args.model_dir, "X-train", "test", args.tail)
            plot_sample_angle(X_train, y_train, X_translate, y_translate, args.model_dir, "X-train", "translate", args.tail)
            plot_sample_angle(Z_train, y_train, Z_test, y_test, args.model_dir, "Z-train", "test", args.tail)
            plot_sample_angle(Z_train, y_train, Z_translate, y_translate, args.model_dir, "Z-train", "translate", args.tail)

        if args.sample_angle_combined:
            plot_sample_angle_combined(X_train, y_train, X_test, y_test, args.model_dir, "X-train", "test", args.tail)
            plot_sample_angle_combined(X_train, y_train, X_translate, y_translate, args.model_dir, "X-train", "translate", args.tail)
            plot_sample_angle_combined(Z_train, y_train, Z_test, y_test, args.model_dir, "Z-train", "test", args.tail)
            plot_sample_angle_combined(Z_train, y_train, Z_translate, y_translate, args.model_dir, "Z-train", "translate", args.tail)
