import os
import numpy as np
import functionals as F
from scipy.special import softmax

class Vector:
    def __init__(self, layers, eta, eps, lmbda=500):
        self.layers = layers
        self.eta = eta
        self.eps = eps
        self.lmbda = lmbda

    def __call__(self, Z, y=None):
        for layer in range(self.layers):
            Z, y_approx = self.forward(layer, Z, y)
            self.arch.update_loss(layer, *self.compute_loss(Z, y_approx))
        return Z
    
    def forward(self, layer, Z, y=None):
        if y is not None:
            self.init(Z, y)
            self.save_weights(layer)
            self.save_gam(layer)
        else:
            self.load_weights(layer)
            self.load_gam(layer)
        expd = Z @ self.E.T
        comp = np.stack([Z @ C.T for C in self.Cs])
        clus, y_approx = self.nonlinear(comp)
        Z = Z + self.eta * (expd - clus)
        Z = F.normalize(Z)
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
        return F.normalize(X)

    def postprocess(self, X):
        return F.normalize(X)

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
        return self.E, self.Cs

    def save_gam(self, layer):
        weight_dir = os.path.join(self.arch.model_dir, "weights")
        os.makedirs(weight_dir, exist_ok=True)
        save_path = os.path.join(weight_dir, f"{self.block_id}_{layer}_gam.npy")
        np.save(save_path, self.gam)

    def load_gam(self, layer):
        weight_dir = os.path.join(self.arch.model_dir, "weights")
        save_path = os.path.join(weight_dir, f"{self.block_id}_{layer}_gam.npy")
        self.gam = np.load(save_path)
        return self.gam
