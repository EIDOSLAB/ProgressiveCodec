
from os.path import join   
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim
import math 
import torch

from collections import OrderedDict
def create_savepath(base_path):
    very_best  = join(base_path,"_very_best.pth.tar")
    last = join(base_path,"_last.pth.tar")
    return last, very_best



