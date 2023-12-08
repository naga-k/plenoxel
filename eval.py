import torch
from render import render_rays
from utils import mse_to_psnr

@torch.no_grad()
def eval_validation(model, dataset, device, hn=0, hf=1., nb_bins=192):
    model.eval()  # Set the model to evaluation mode
    val_loss = 0.0
    psnr = 0.0
    for i in range(len(dataset)):
        batch = dataset[i]
        rays = batch['rays'].to(device)        
        ground_truth_px_values = batch['rgbs'].to(device)
        ray_origins, ray_directions = rays[:, :3], rays[:, 3:6]

        regenerated_px_values = render_rays(model, ray_origins, ray_directions, hn= hn, hf=hf, nb_bins=192)
        loss = torch.nn.functional.mse_loss(regenerated_px_values, ground_truth_px_values)
        val_loss += loss.item()
        psnr += mse_to_psnr(loss.item())

    avg_val_loss = val_loss / len(dataset)
    avg_psnr = psnr / len(dataset)
    
    return avg_val_loss, avg_psnr