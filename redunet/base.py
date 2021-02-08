import os
import numpy as np
from scipy.special import softmax


class Architecture:
    def __init__(self, blocks, model_dir, num_classes, batch_size=1000, save_loss=True, save_layers=[], sgd=False):
        self.blocks = blocks
        self.model_dir = model_dir
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.save_loss = save_loss
        self.save_layers = save_layers
        self.sgd = sgd
        
    def __call__(self, Z, y=None):
        for b, block in enumerate(self.blocks):
            block.load_arch(self, b)
            self.init_loss()

            Z = block.preprocess(Z)
            Z = block(Z, y, self.sgd)
            Z = block.postprocess(Z)
        return Z

    def __getitem__(self, i):
        return self.blocks[i]

    def init_loss(self):
        self.loss_dict = {"loss_total": [],
                          "loss_expd": [], 
                          "loss_comp": []}

    def update_loss(self, layer, loss_total, loss_expd, loss_comp):
        self.loss_dict["loss_total"].append(loss_total)
        self.loss_dict["loss_expd"].append(loss_expd)
        self.loss_dict["loss_comp"].append(loss_comp)
        print(f"layer: {layer} | loss_total: {loss_total:5f} | loss_expd: {loss_expd:5f} | loss_comp: {loss_comp:5f}")
        
