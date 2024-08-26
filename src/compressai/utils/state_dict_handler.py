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

def adapt_state_dict(old_state_dict,mode = "refine", old = False):

    new_ordered_dict = OrderedDict()
    if mode == "refine":
        for key, value in old_state_dict.items():

            nm_base = "base_net." if old is False else ""

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
                new_key =  nm_base + key
                new_ordered_dict[new_key] = value   
        return new_ordered_dict 
    else:
        for key, value in old_state_dict.items():

            nm_base = "base_net." if old is False else ""

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

                new_key =  nm_base + key
                new_ordered_dict[new_key] = value   
        return new_ordered_dict      
    


def complete_args(new_args):
    new_args.multiple_encoder = False if "multiple_encoder" not in new_args else new_args.multiple_encoder
    new_args.multiple_hyperprior = False if "multiple_hyperprior" not in new_args else new_args.multiple_hyperprior
    new_args.delta_encode = False  if "delta_encode" not in new_args else new_args.delta_encode
    new_args.residual_before_lrp = False if "residual_before_lrp" not in new_args else new_args.residual_before_lrp
    new_args.double_dim = False if "double_dim" not in new_args else new_args.double_dim
    return new_args