from itertools import product
import time
import numpy as np

from .vector import Vector
import functionals as F

class Fourier2D(Vector):
    def __init__(self, layers, eta, eps, lmbda=5000):
        super().__init__(layers, eta, eps, lmbda)

    def forward(self, layer, V, y=None, init=False):
        if init:
            self.init(V, y)
            self.save_weights(layer)
        else:
            self.load_weights(layer)
        expd = np.einsum("bi...,ih...->bh...", V, self.E.conj(), optimize=True)
        comp = np.stack([np.einsum("bi...,ih...->bh...", V, C_j.conj(), optimize=True) \
                for C_j in self.Cs])
        clus, y_approx = self.nonlinear(comp)
        V = V + self.eta * (expd - clus)
        V = F.normalize(V)
        return V, y_approx

    def compute_E(self, V):
        m, C, H, W = V.shape
        alpha = C / (m * self.eps)
        pre_inv = alpha * F.batch_cov(V, self.arch.batch_size) \
                  + np.eye(C)[..., np.newaxis, np.newaxis]
        E = np.empty_like(pre_inv).astype(np.complex)
        for h, w in product(range(H), range(W)):
            E[:, :, h, w] = alpha * np.linalg.inv(pre_inv[:, :, h, w])
        self.E = E

    def compute_Cs(self, V, y):
        m, C, H, W = V.shape
        Cs = np.empty((self.num_classes, C, C, H, W), dtype=np.complex)
        for j in np.arange(self.num_classes):
            V_j = V[y==j]
            m_j = V_j.shape[0]
            if m_j == 0:
                continue
            alpha_j = C / (m_j * self.eps)
            pre_inv = alpha_j * F.batch_cov(V_j, self.arch.batch_size) \
                + np.eye(C)[..., np.newaxis, np.newaxis]
            for h, w in product(range(H), range(W)):
                Cs[j, :, :, h, w] =  alpha_j * np.linalg.inv(pre_inv[:, :, h, w])
        self.Cs = Cs

    def compute_loss(self, V, y):
        m, C, H, W = V.shape
        alpha = C / (m * self.eps)
        cov = alpha * F.batch_cov(V, self.arch.batch_size) \
                + np.eye(C)[..., np.newaxis, np.newaxis]
        loss_expd = np.sum([np.linalg.slogdet(cov[:, :, h, w])[1] for h, w in product(range(H), range(W))]) / (2 * H * W)

        loss_comp = 0.
        Cs = np.empty((self.num_classes, C, C, H, W), dtype=np.complex)
        for j in range(self.num_classes):
            V_j = V[y==int(j)]
            m_j = V_j.shape[0]
            if m_j == 0:
                continue
            alpha_j = C / (m_j * self.eps) 
            cov_j = alpha_j * F.batch_cov(V_j, self.arch.batch_size) \
                        + np.eye(C)[..., np.newaxis, np.newaxis]
            loss_comp += m_j / m * np.sum([np.linalg.slogdet(cov_j[:, :, h, w])[1] for h, w in product(range(H), range(W))]) / (2 * H * W)
        return loss_expd - loss_comp, loss_expd, loss_comp

    def preprocess(self, X):
        Z = F.normalize(X)
        return np.fft.fft2(X, norm='ortho', axes=(2, 3))

    def postprocess(self, X):
        Z = np.fft.ifft2(X, norm='ortho', axes=(2, 3))
        return F.normalize(Z)
