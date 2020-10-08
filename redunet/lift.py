import torch
import torch.nn.functional as F


class Lift():
    def __init__(self, kernels, stride=1, relu=True):
        self.kernels = kernels
        self.stride = stride
        self.relu = relu

    def load_arch(self, arch, block_id):
        pass
    
    def init(self, Z, y):
        return self(Z)
    
    def preprocess(self, X):
        return X
    
    def postprocess(self, X):
        return X


class Lift1D(Lift):
    def __init__(self, kernels, stride=1, relu=True):
        assert len(kernels.shape) == 3, "kernel should have dimensions (out_channel, in_channel, kernel_size)"
        super(Lift1D, self).__init__(kernels, stride, relu)
    
    def __call__(self, Z, y=None):
        ksize = self.kernels.shape[2]
        _Z = F.pad(torch.tensor(Z).float(), (0, ksize-1), 'circular')
        _kernels = torch.tensor(self.kernels).float()
        out = F.conv1d(_Z, _kernels, stride=self.stride)
        if self.relu:
            out = F.relu(out)
        return out.numpy()


class Lift2D(Lift):
    def __init__(self, kernels, stride=1, relu=True):
        assert len(kernels.shape) == 4, "kernel should have dimensions " \
            "(out_channel, in_channel, kernel_height, kernel_width)"
        super(Lift2D, self).__init__(kernels, stride, relu)
    
    def __call__(self, Z, y=None):
        ksize = self.kernels.shape[2]
        _Z = F.pad(torch.tensor(Z).float(), (0, ksize-1, 0, ksize-1), 'circular')
        _kernels = torch.tensor(self.kernels).float()
        out = F.conv2d(_Z, _kernels, stride=self.stride)
        if self.relu:
            out = F.relu(out)
        return out.numpy()