
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




from os.path import join   
from PIL import Image
from torchvision import transforms
from pytorch_msssim import ms_ssim
import math 
import torch

from collections import OrderedDict

def replace_keys(checkpoint, multiple_encoder):
    # Creiamo un nuovo OrderedDict con le chiavi modificate all'interno di un ciclo for
    nuovo_ordered_dict = OrderedDict()
    for chiave, valore in checkpoint.items():
        if multiple_encoder:
            if "g_a_enh." in chiave: 
                
                nuova_chiave = chiave.replace("g_a_enh.", "g_a.1.")
                nuovo_ordered_dict[nuova_chiave] = valore
            elif "g_a." in chiave and "g_a.0.1.beta" not in list(checkpoint.keys()): 
                nuova_chiave = chiave.replace("g_a.", "g_a.0.")
                nuovo_ordered_dict[nuova_chiave] = valore  
            else: 
                nuovo_ordered_dict[chiave] = valore   
        else:
            nuovo_ordered_dict[chiave] = valore  
    return nuovo_ordered_dict  

def adapt_state_dict(old_state_dict,args, old = False):

    new_ordered_dict = OrderedDict()
    if args.modality == "refine":
        for key, value in old_state_dict.items():

            nm_base = "base_net." if old is False else "base_net."

            if "g_s.1." in key:
                new_key = nm_base +  key.replace("g_s.1.", "g_s.1.")
                new_ordered_dict[new_key] = value
                new_key = nm_base +  key.replace("g_s.1.", "g_s.2.")
                new_ordered_dict[new_key] = value
                new_key = nm_base +  key.replace("g_s.1.", "g_s.3.")
                new_ordered_dict[new_key] = value
                new_key = nm_base +  key.replace("g_s.1.", "g_s.4.")
                new_ordered_dict[new_key] = value

            else: 
                if "gaussian_conditional" in key or "entropy_bottleneck" in key:
                    new_key = "base_net." + key
                    new_ordered_dict[new_key] = value
                else:
                    new_key =  nm_base + key
                    new_ordered_dict[new_key] = value   
        return new_ordered_dict 
    else:
        for key, value in old_state_dict.items():

            nm_base = "base_net." if old is False else "base_net."

            if "g_s.1.1" in key:
                new_key = nm_base +  key.replace("g_s.1.1.", "g_s.1.1.original_model_layer.")
                new_ordered_dict[new_key] = value
            elif "g_s.1.3" in key:
                new_key =nm_base + key.replace("g_s.1.3.", "g_s.1.3.original_model_layer.")
                new_ordered_dict[new_key] = value
            elif "g_s.1.6" in key:
                new_key = nm_base + key.replace("g_s.1.6.", "g_s.1.6.original_model_layer.")
                new_ordered_dict[new_key] = value
            elif "g_s.1.8" in key:
                new_key = nm_base + key.replace("g_s.1.8.", "g_s.1.8.original_model_layer.")#ddd
                new_ordered_dict[new_key] = value
            else: 
                if "gaussian_conditional" in key or "entropy_bottleneck" in key:
                    new_key = "base_net." + key
                    new_ordered_dict[new_key] = value
                else:
                    new_key =  nm_base + key
                    new_ordered_dict[new_key] = value   
        return new_ordered_dict      


def read_image(filepath,):
    #assert filepath.is_file()
    img = Image.open(filepath)   
    img = img.convert("RGB")
    return transforms.ToTensor()(img) 

class AverageMeter:
    """Compute running average."""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count






def create_savepath(base_path):
    very_best  = join(base_path,"_very_best.pth.tar")
    last = join(base_path,"_last.pth.tar")
    return last, very_best








def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


