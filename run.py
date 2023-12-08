from model import PlenoxelModel
import torch
from torch.utils.data import DataLoader
from train import train
from test import test # type: ignore
from datasets import BlenderDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    root_dir = '/mmfs1/project/jacobcha/nk643/data_src/nerf_synthetic/'
    nerf_data = 'lego'
    img_wh = (800, 800)

    training_dataset = BlenderDataset(root_dir=root_dir+nerf_data, split='train', img_wh=img_wh)
    data_loader = DataLoader(training_dataset, batch_size=65536, shuffle=True, num_workers=8)

    model = PlenoxelModel(N=256).to(device)
    model_optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(model_optimizer, milestones=[2, 4, 8], gamma=0.5)
    
    train(model, model_optimizer, scheduler, data_loader, \
        root_dir = '/project/jacobcha/nk643/data_src/nerf_synthetic/' , \
        nerf_data = 'lego', img_wh = (800,800), \
        device=device, hn=2, hf=6, nb_epochs= 20, \
        nb_bins=192, eval = True)

    torch.save(model.state_dict(), f'/project/jacobcha/nk643/plenoxel/checkpoints/{nerf_data}.pth')