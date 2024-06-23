import os
from PIL import Image
from torchvision import transforms
import torch
from losses.ssim_loss import _ssim, create_window
from functools import partial

def get_pos_sample(train_img_root, device, batch_size):
    pos_img_name = train_img_root
    transforms_x = transforms.Compose([transforms.Resize(256, Image.ANTIALIAS),
                                       transforms.ToTensor()])
    pos_img = transforms_x(Image.open(pos_img_name).convert('RGB'))
    pos_img = [pos_img.unsqueeze(0)]*batch_size
    pos_img = torch.cat(pos_img, dim=0)
    pos_img = pos_img.to(device)
    return pos_img