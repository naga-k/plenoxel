
from torch import nn
import torch
from torch.nn import functional as F

from eval_spherical import eval_sphericals_function

class PlenoxelModel(nn.Module):
    def __init__(self, N=256, scale=1):
        """
        Initialize the PlenoxelModel class.

        Args:
            N (int): The size of the voxel grid (default is 256).
            scale (float): The scaling factor for the voxel grid (default is 1.5).
        """
        super(PlenoxelModel, self).__init__()

        # Initialize the voxel grid with ones divided by 100
        # Each voxel stores 27 + 1 values corresponding to the spherical harmonics coefficients (colors 9 * 3 + 1)
        self.voxel_grid = nn.Parameter(torch.ones((N, N, N, 27 + 1)) / 100)
        self.N = N
        self.scale = scale

    def forward(self, x, d):
        color = torch.zeros_like(x)
        sigma = torch.zeros((x.shape[0]), device = x.device)
        mask = (x[:,0].abs() < self.scale) & (x[:, 1].abs() < self.scale) & (x[:, 2].abs() < self.scale)

        idx = (x[mask] / (2 * self.scale / self.N) + self.N/2 ).long().clip(0, self.N - 1)
        tmp = self.voxel_grid[idx[:, 0], idx[:, 1], idx[:, 2]]
        sigma[mask],k = F.relu(tmp[:, 0]), tmp[:, 1:]
        color[mask] = eval_sphericals_function(k.reshape(-1, 3, 9), d[mask])
        return color, sigma