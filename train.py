from tqdm import tqdm
import torch.nn.functional as F
from render import render_rays
from datasets import BlenderDataset
from eval import eval_validation
from torch.utils.tensorboard import SummaryWriter 


writer = SummaryWriter(log_dir='/project/jacobcha/nk643/plenoxel/logs')

def train(model, optimizer, scheduler, data_loader, \
        root_dir = '/project/jacobcha/nk643/data_src/nerf_synthetic' , \
        nerf_data = 'lego', img_wh = (400,400), \
        device='cpu', hn=0, hf=1, nb_epochs=int(1e5), \
        nb_bins=192, eval = False):
    training_loss = []
    eval_dataset = None
    if eval:
        eval_dataset = BlenderDataset(root_dir=root_dir + nerf_data, split='val', img_wh=img_wh)
        
    for epoch in range(nb_epochs):
        for batch in tqdm(data_loader):
            rays = batch['rays'].to(device)  # Adjusted line
            ground_truth_px_values = batch['rgbs'].to(device)  # Adjusted line
            ray_origins, ray_directions = rays[:, :3], rays[:, 3:6]  # Adjusted line

            regenerated_px_values = render_rays(model, ray_origins, ray_directions, hn=hn, hf=hf, nb_bins=nb_bins)
            loss = F.mse_loss(ground_truth_px_values, regenerated_px_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            training_loss.append(loss.item())

        scheduler.step()

        

        if eval:
            avg_val_loss, avg_psnr = eval_validation(model, eval_dataset, device,hn=hn, hf=hf, nb_bins=nb_bins)
            print(f'Average Validation Loss: {avg_val_loss:.4f}, Average PSNR: {avg_psnr:.4f}')
            writer.add_scalar('Loss/Validation', avg_val_loss, epoch)
            writer.add_scalar('PSNR/Validation', avg_psnr, epoch)
            model.train()

    writer.close()

    return training_loss