from datasets import BlenderDataset
from test import test # type: ignore
import torch
from model import PlenoxelModel

if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PlenoxelModel(N=256).to(device)
    model.load_state_dict(torch.load('/mmfs1/project/jacobcha/nk643/plenoxel/checkpoints/lego.pth'))
    model.eval()
    root_dir = '/mmfs1/project/jacobcha/nk643/data_src/nerf_synthetic/'
    nerf_data = 'lego'
    testing_dataset = BlenderDataset(root_dir=root_dir+nerf_data, split='test', img_wh=(800, 800))
    for img_index in range(len(testing_dataset)):
        test(2, 6, model, testing_dataset, img_index=img_index, nb_bins=192, H=800, W=800)