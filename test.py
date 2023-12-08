import numpy as np
import torch
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

from render import render_rays


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

@torch.no_grad()
def test(hn, hf, model, dataset, chunk_size=2048, img_index=0, nb_bins=192, H=400, W=400):
    rays = dataset[img_index]['rays'].to(device)
    ground_truth_img = dataset[img_index]['rgbs'].to(device).reshape(H, W, 3).cpu().numpy()
    ray_origins, ray_directions = rays[:, :3], rays[:, 3:6]  

    data = []
    for i in range(int(np.ceil(H / chunk_size))):
        ray_origins_ = ray_origins[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        ray_directions_ = ray_directions[i * W * chunk_size: (i + 1) * W * chunk_size].to(device)
        regenerated_px_values = render_rays(model, ray_origins_, ray_directions_, hn=hn, hf=hf, nb_bins=nb_bins)
        data.append(regenerated_px_values)
    img = torch.cat(data).data.cpu().numpy().reshape(H, W, 3)

    psnr = compare_psnr(ground_truth_img, img)

    print(f'PSNR for image {img_index}: {psnr:.2f} dB')

    plt.figure()
    plt.axis('off')
    plt.imshow(img)
    plt.show()
    plt.savefig(f'/mmfs1/project/jacobcha/nk643/plenoxel/Imgs/img_{img_index}.png', bbox_inches='tight')
    plt.close()