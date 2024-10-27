import wandb
import random
import sys
import numpy as np
from compressai.utils.functions import  create_savepath
from compressai.utils.parser import parse_args
from compressai.utils.plot import plot_rate_distorsion
from compressai.utils.state_dict_handler import complete_args, adapt_state_dict, replace_keys
import time
import shutil
import torch
import math
import torch.nn as nn
import torch.optim as optim
from   compressai.training.step import train_one_epoch, valid_epoch, test_epoch, compress_with_ac
from torch.utils.data import DataLoader
from torchvision import transforms
from compressai.training.loss import ScalableRateDistortionLoss, DistortionLoss
from compressai.datasets.utils import ImageFolder, TestKodakDataset
from compressai.models import get_model
#from compress.zoo import models
from compressai.utils.result_list import tri_planet_22_bpp, tri_planet_22_psnr, tri_planet_23_bpp, tri_planet_23_psnr
import os
from collections import OrderedDict


def initialize_model_from_pretrained( checkpoint,
                                     args,
                                     checkpoint_enh = None):

    sotto_ordered_dict = OrderedDict()

    for c in list(checkpoint.keys()):
        if "g_s" in c: 
            if args.multiple_decoder:
                nuova_stringa = "g_s.0." + c[4:]
                sotto_ordered_dict[nuova_stringa] = checkpoint[c]
            else:
                sotto_ordered_dict[c] = checkpoint[c]

        elif "g_a" in c: 
            if args.multiple_encoder:
                nuova_stringa = "g_a.0." + c[4:]
                sotto_ordered_dict[nuova_stringa] = checkpoint[c]
            else:
                sotto_ordered_dict[c] = checkpoint[c]
        elif "cc_" in c or "lrp_" in c or "gaussian_conditional" or "entropy_bottleneck" in c: 
            sotto_ordered_dict[c] = checkpoint[c]
        else:
            continue


    for c in list(sotto_ordered_dict.keys()):
        if "h_scale_s" in c or "h_a" in c  or "h_mean_s" in c:
            sotto_ordered_dict.pop(c)
    if args.multiple_hyperprior:
        for c in list(checkpoint.keys()):
            if "h_mean_s" in c:
                nuova_stringa = "h_mean_s.0." + c[9:]
                sotto_ordered_dict[nuova_stringa] = checkpoint[c]     
            elif "h_scale_s" in c:
                nuova_stringa = "h_scale_s.0." + c[10:]
                sotto_ordered_dict[nuova_stringa] = checkpoint[c]   


    for c in list(sotto_ordered_dict.keys()):
        if "h_a" in c:
            sotto_ordered_dict.pop(c)


    if checkpoint_enh is not None:
        print("prendo anche il secondo modello enhanced")
        for c in list(checkpoint_enh.keys()):
            if "g_s" in c: 
                nuova_stringa = "g_s.1." + c[4:]
                sotto_ordered_dict[nuova_stringa] = checkpoint_enh[c]

            else:
                continue




    return sotto_ordered_dict
    


def sec_to_hours(seconds):
    a=str(seconds//3600)
    b=str((seconds%3600)//60)
    c=str((seconds%3600)%60)
    d=["{} hours {} mins {} seconds".format(a, b, c)]
    print(d[0])




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


class CustomDataParallel(nn.DataParallel):
    """Custom DataParallel to access the module methods."""

    def __getattr__(self, key):
        try:
            return super().__getattr__(key)
        except AttributeError:
            return getattr(self.module, key)


def configure_optimizers(net, args):
    """Separate parameters for the main optimizer and the auxiliary optimizer.
    Return two optimizers"""

    parameters = {
        n
        for n, p in net.named_parameters()
        if not n.endswith(".quantiles") and p.requires_grad
    }
    aux_parameters = {
        n
        for n, p in net.named_parameters()
        if n.endswith(".quantiles") and p.requires_grad
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = parameters & aux_parameters
    union_params = parameters | aux_parameters

    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    optimizer = optim.Adam(
        (params_dict[n] for n in sorted(parameters)),
        lr=args.learning_rate,
    )
    aux_optimizer = optim.Adam(
        (params_dict[n] for n in sorted(aux_parameters)),
        lr=args.aux_learning_rate,
    )
    return optimizer, aux_optimizer


def train_one_epoch(
    model, criterion, train_dataloader, optimizer, aux_optimizer, epoch, clip_max_norm
):
    model.train()
    device = next(model.parameters()).device

    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        aux_optimizer.zero_grad()

        out_net = model(d)

        out_criterion = criterion(out_net, d)
        out_criterion["loss"].backward()
        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()

        aux_loss = model.aux_loss()
        aux_loss.backward()
        aux_optimizer.step()

        if i % 100 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].item() * 255 ** 2 / 3:.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
                f"\tAux loss: {aux_loss.item():.2f}"
            )





def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename[:-8]+"_best"+filename[-8:])




def main(argv):

    args = parse_args(argv)
    print(args)

    

    openimages_path = "/scratch/dataset/openimages"
    kodak_path = "/scratch/dataset/kodak"
    save_path = "/scratch/ResDSIC/models/"


    dict_base_model = {"q2":"/scratch/base_devil/weights/q2/model.pth",
                        "q5":"/scratch/base_devil/weights/q5/model.pth"}

    
    if  args.model == "multidec": 
        wandb.init( config= args, project="ResDSIC-refine", entity="albipresta") #ddd
    elif len(args.lmbda_list) > 2:
        wandb.init( config= args, project="ResDSIC-3L", entity="albipresta")
    elif args.joiner_policy == "cond":
        wandb.init( config= args, project="ResDSIC-cond", entity="albipresta") 
    else:
        wandb.init( config= args, project="ResDSIC-Delta-Unet", entity="albipresta")  #dddd dddd ddddd
    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
    

    print("initialize dataset")
    
    train_transforms = transforms.Compose(
        [transforms.RandomCrop(args.patch_size), transforms.ToTensor()]
    )

    test_transforms = transforms.Compose(
        [ transforms.ToTensor()]
    )

    train_dataset = ImageFolder(openimages_path, split="train", transform=train_transforms, num_images=args.num_images)
    valid_dataset = ImageFolder(openimages_path, split="test", transform=train_transforms, num_images=args.num_images_val)
    test_dataset = TestKodakDataset(data_dir=kodak_path, transform= test_transforms)

    filelist = test_dataset.image_path

    device = "cuda" 

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        pin_memory=(device == "cuda"),
    )

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.valid_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )

    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers,shuffle=False,pin_memory=(device == "cuda"),)

    if args.lmbda_list == []:
        lmbda_list = None
    else:
        lmbda_list = args.lmbda_list

    
    if args.checkpoint == "none":
        net = get_model(args,device, lmbda_list)
        net = net.to(device)
        net.update()

    else:
        checkpoint = torch.load(save_path + "/" + args.checkpoint + "/" + args.name_model, map_location=device)
        new_args = checkpoint["args"]
        new_args = complete_args(new_args)
        lmbda_list = new_args.lmbda_list
        checkpoint_model = replace_keys(checkpoint["state_dict"], multiple_encoder=new_args.multiple_encoder)
        net = get_model(new_args,device, lmbda_list)

        net.load_state_dict(checkpoint_model,strict = True)

        net.update()

    if args.checkpoint_base != "none":
        print("entro qua per il checkpoint base")
        checkpoin_base_model = dict_base_model[args.checkpoint_base]
        base_checkpoint = torch.load(checkpoin_base_model,map_location=device)
        new_check = initialize_model_from_pretrained(base_checkpoint, args)
        net.load_state_dict(new_check,strict = False)
        net.update() 
        if args.freeze_base:
            net.freeze_base_net(args.multiple_hyperprior)   



    if args.cuda and torch.cuda.device_count() > 1:
        net = CustomDataParallel(net)

    last_epoch = 0

    optimizer, aux_optimizer = configure_optimizers(net, args)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", factor=0.3, patience=args.patience)
    


    criterion = ScalableRateDistortionLoss(lmbda_list=args.lmbda_list)
    #criterion = DistortionLoss() #if args.model == "multidec" #else  ScalableRateDistortionLoss(lmbda_list=args.lmbda_list)
    

    best_loss = float("inf")
    counter = 0
    epoch_enc = 0





    list_quality = [0,10]
    
    
    print("PRIMA DI COMINCIARE IL TRAINING, FACCIAMO IL PRIMO TEST INIZIALE")
    if args.checkpoint != "none":
        pr_list = [0,0.05,0.1,0.25,0.5,0.6,0.75,1,1.25,2,3,5,10] #ggg
        mask_pol = "point-based-std"
    
        bpp_init, psnr_init,_ = compress_with_ac(net, #net 
                                            filelist, 
                                            device,
                                           epoch = -10, 
                                           pr_list =pr_list,  
                                            mask_pol = mask_pol,
                                            writing = None)
    
        print("----> ",bpp_init," ",psnr_init)
    


    for epoch in range(last_epoch, args.epochs):
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        train_one_epoch(
            net,
            criterion,
            train_dataloader,
            optimizer,
            aux_optimizer,
            epoch,
            args.clip_max_norm,
        )
        loss = test_epoch(epoch, test_dataloader, net, criterion)
        lr_scheduler.step(loss)

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if args.save:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "loss": loss,
                    "optimizer": optimizer.state_dict(),
                    "aux_optimizer": aux_optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                },
                is_best,
                args.save_path,
            )


if __name__ == "__main__":
    main(sys.argv[1:])
