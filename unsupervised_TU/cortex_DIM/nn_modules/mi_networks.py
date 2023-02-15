"""Module for networks used for computing MI.

"""

import numpy as np
import torch
import torch.nn as nn

class View(torch.nn.Module):
    """Basic reshape module.
    """
    def __init__(self, *shape):
        """
        Args:
            *shape: Input shape.
        """
        super().__init__()
        self.shape = shape

    def forward(self, input):
        """Reshapes tensor.
        Args:
            input: Input tensor.
        Returns:
            torch.Tensor: Flattened tensor.
        """
        return input.view(*self.shape)


class Unfold(torch.nn.Module):
    """Module for unfolding tensor.
    Performs strided crops on 2d (image) tensors. Stride is assumed to be half the crop size.
    """
    def __init__(self, img_size, fold_size):
        """
        Args:
            img_size: Input size.
            fold_size: Crop size.
        """
        super().__init__()

        fold_stride = fold_size // 2
        self.fold_size = fold_size
        self.fold_stride = fold_stride
        self.n_locs = 2 * (img_size // fold_size) - 1
        self.unfold = torch.nn.Unfold((self.fold_size, self.fold_size),
                                      stride=(self.fold_stride, self.fold_stride))

    def forward(self, x):
        """Unfolds tensor.
        Args:
            x: Input tensor.
        Returns:
            torch.Tensor: Unfolded tensor.
        """
        N = x.size(0)
        x = self.unfold(x).reshape(N, -1, self.fold_size, self.fold_size, self.n_locs * self.n_locs)\
            .permute(0, 4, 1, 2, 3)\
            .reshape(N * self.n_locs * self.n_locs, -1, self.fold_size, self.fold_size)
        return x


class Fold(torch.nn.Module):
    """Module (re)folding tensor.
    Undoes the strided crops above. Works only on 1x1.
    """
    def __init__(self, img_size, fold_size):
        """
        Args:
            img_size: Images size.
            fold_size: Crop size.
        """
        super().__init__()
        self.n_locs = 2 * (img_size // fold_size) - 1

    def forward(self, x):
        """(Re)folds tensor.
        Args:
            x: Input tensor.
        Returns:
            torch.Tensor: Refolded tensor.
        """
        dim_c, dim_x, dim_y = x.size()[1:]
        x = x.reshape(-1, self.n_locs * self.n_locs, dim_c, dim_x * dim_y)
        x = x.reshape(-1, self.n_locs * self.n_locs, dim_c, dim_x * dim_y)\
            .permute(0, 2, 3, 1)\
            .reshape(-1, dim_c * dim_x * dim_y, self.n_locs, self.n_locs).contiguous()
        return x


class Permute(torch.nn.Module):
    """Module for permuting axes.
    """
    def __init__(self, *perm):
        """
        Args:
            *perm: Permute axes.
        """
        super().__init__()
        self.perm = perm

    def forward(self, input):
        """Permutes axes of tensor.
        Args:
            input: Input tensor.
        Returns:
            torch.Tensor: permuted tensor.
        """
        return input.permute(*self.perm)

class MIFCNet(nn.Module):
    """Simple custom network for computing MI.

    """
    def __init__(self, n_input, n_units):
        """

        Args:
            n_input: Number of input units.
            n_units: Number of output units.
        """
        super().__init__()

        assert(n_units >= n_input)

        self.linear_shortcut = nn.Linear(n_input, n_units)
        self.block_nonlinear = nn.Sequential(
            nn.Linear(n_input, n_units),
            nn.BatchNorm1d(n_units),
            nn.ReLU(),
            nn.Linear(n_units, n_units)
        )

        # initialize the initial projection to a sort of noisy copy
        eye_mask = np.zeros((n_units, n_input), dtype=np.uint8)
        for i in range(n_input):
            eye_mask[i, i] = 1

        self.linear_shortcut.weight.data.uniform_(-0.01, 0.01)
        self.linear_shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)

    def forward(self, x):
        """

        Args:
            x: Input tensor.

        Returns:
            torch.Tensor: network output.

        """
        h = self.block_nonlinear(x) + self.linear_shortcut(x)
        return h


class MI1x1ConvNet(nn.Module):
    """Simple custorm 1x1 convnet.

    """
    def __init__(self, n_input, n_units):
        """

        Args:
            n_input: Number of input units.
            n_units: Number of output units.
        """

        super().__init__()

        self.block_nonlinear = nn.Sequential(
            nn.Conv1d(n_input, n_units, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(n_units),
            nn.ReLU(),
            nn.Conv1d(n_units, n_units, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.block_ln = nn.Sequential(
            Permute(0, 2, 1),
            nn.LayerNorm(n_units),
            Permute(0, 2, 1)
        )

        self.linear_shortcut = nn.Conv1d(n_input, n_units, kernel_size=1,
                                         stride=1, padding=0, bias=False)

        # initialize shortcut to be like identity (if possible)
        if n_units >= n_input:
            eye_mask = np.zeros((n_units, n_input, 1), dtype=np.uint8)
            for i in range(n_input):
                eye_mask[i, i, 0] = 1
            self.linear_shortcut.weight.data.uniform_(-0.01, 0.01)
            self.linear_shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)

    def forward(self, x):
        """

            Args:
                x: Input tensor.

            Returns:
                torch.Tensor: network output.

        """
        h = self.block_ln(self.block_nonlinear(x) + self.linear_shortcut(x))
        return h
