import torch 
import numpy as np

def compute_accumulated_transmittance(alphas):
    accumulated_transmittance = torch.cumprod(alphas, 1)
    return torch.cat((torch.ones((accumulated_transmittance.shape[0], 1), device=alphas.device),
                      accumulated_transmittance[:, :-1]), dim=-1)

def mse_to_psnr(mse):
    return 20 * np.log10(1.0) - 10 * np.log10(mse)