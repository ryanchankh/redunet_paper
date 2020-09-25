
import os
import numpy as np
import train_func as tf
from scipy.special import softmax

class Vector:
    def __init__(self, layers, eta, eps, lmbda=5000):
        self.layers = layers
        self.eta = eta
        self.eps = eps
        self.lmbda = lmbda

    def __call__(self, Z, y=None):
        init = y is not None
        for layer in range(self.layers):
            Z, y_approx = self.forward(layer, Z, y, init)

            if self.arch.save_loss:
                self.arch.update_loss(layer, *self.compute_loss(Z, y_approx))
            if init and layer in self.arch.save_layers:
                self.save_layer(layer, Z)
        return Z

    def forward(self, layer, Z, y=None, init=False):
        if init:
            self.init(Z, y)
            self.save_weights(layer)
        else:
            self.load_weights(layer)
        expd = Z @ self.E.T
        comp = np.stack([Z @ C.T for C in self.Cs])
        clus, y_approx = self.nonlinear(comp)
        Z = Z + self.eta * (expd - clus)
        Z = tf.normalize(Z)
        return Z, y_approx

    def load_arch(self, arch, block_id):
        self.arch = arch
        self.block_id = block_id
        self.num_classes = self.arch.num_classes

    def init(self, Z, y):
        self.compute_gam(y)
        self.compute_E(Z)
        self.compute_Cs(Z, y)

    def compute_gam(self, y):
        m_j = [(y==j).nonzero()[0].size for j in range(self.num_classes)]
        self.gam = np.array(m_j) / y.size

    def compute_E(self, X):
        m, d = X.shape
        Z = X.T
        I = np.eye(d)
        c = d / (m * self.eps)
        E = c * np.linalg.inv(I + c * Z @ Z.T)
        self.E = E

    def compute_Cs(self, X, y):
        m, d = X.shape
        Z = X.T
        I = np.eye(d)
        Cs = np.empty((self.num_classes, d, d))
        for j in range(self.num_classes):
            idx = (y == int(j))
            Z_j = Z[:, idx]
            m_j = Z_j.shape[0]
            c_j = d / (m_j * self.eps)
            C = c_j * np.linalg.inv(I + c_j * Z_j @ Z_j.T)
            Cs[j] = C
        self.Cs = Cs

    def compute_loss(self, Z, y):
        m, d = Z.shape
        I = np.eye(d)
        
        c = d / (m * self.eps)
        logdet = np.linalg.slogdet(I + c * Z.T @ Z)[1]
        loss_expd = logdet / 2.

        loss_comp = 0.
        for j in np.arange(self.num_classes):
            idx = (y == int(j))
            Z_j = Z[idx, :]
            m_j = Z_j.shape[0]
            if m_j == 0:
                continue
            c_j = d / (m_j * self.eps)
            logdet_j = np.linalg.slogdet(I + c_j * Z_j.T @ Z_j)[1]
            loss_comp += self.gam[j] * logdet_j / 2.
        return loss_expd - loss_comp, loss_expd, loss_comp

    def preprocess(self, X):
        m = X.shape[0]
        X = X.reshape(m, -1)
        return tf.normalize(X)

    def postprocess(self, X):
        return tf.normalize(X)

    def nonlinear(self, Bz):
        axes = tuple(np.arange(2, len(Bz.shape)))
        norm = np.linalg.norm(Bz.reshape(Bz.shape[0], Bz.shape[1], -1), axis=2)
        norm = np.clip(norm, 1e-8, norm)
        pred = softmax(-self.lmbda * norm, axis=0)
        y = np.argmax(pred, axis=0)
        gam = np.expand_dims(self.gam, tuple(np.arange(1, len(Bz.shape))))
        out = np.sum(gam * Bz * np.expand_dims(pred, axes), axis=0)
        return out, y

    def save_weights(self, layer):
        weights = np.vstack([
                    self.E[np.newaxis, :],
                    self.Cs
                  ])
        weight_dir = os.path.join(self.arch.model_dir, "weights")
        os.makedirs(weight_dir, exist_ok=True)
        save_path = os.path.join(weight_dir, f"{self.block_id}_{layer}.npy")
        np.save(save_path, weights)

    def load_weights(self, layer):
        weight_dir = os.path.join(self.arch.model_dir, "weights")
        save_path = os.path.join(weight_dir, f"{self.block_id}_{layer}.npy")
        weights = np.load(save_path)
        self.E = weights[0]
        self.Cs = weights[1:]
        return self

    def save_layer(self, layer, Z):
        layer_dir = os.path.join(self.arch.model_dir, "features", "layers")
        os.makedirs(layer_dir, exist_ok=True)
        np.save(os.path.join(layer_dir, f"block{self.block_id}_layer{layer}.npy"), Z)
