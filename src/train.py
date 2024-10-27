import wandb
import random
import sys
import numpy as np
from compress.utils.functions import  create_savepath
from compress.utils.parser import parse_args
from compress.utils.plot import plot_rate_distorsion
from compress.utils.state_dict_handler import complete_args, adapt_state_dict, replace_keys
import time
import shutil
import torch
import math
import torch.nn as nn
import torch.optim as optim
from   compress.training.step import train_one_epoch, valid_epoch, test_epoch, compress_with_ac
from torch.utils.data import DataLoader
from torchvision import transforms
from compress.training.loss import ScalableRateDistortionLoss, DistortionLoss
from compress.datasets.utils import ImageFolder, TestKodakDataset
from compress.models import get_model
#from compress.zoo import models
from compress.utils.result_list import tri_planet_22_bpp, tri_planet_22_psnr, tri_planet_23_bpp, tri_planet_23_psnr
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






def save_checkpoint(state, is_best, last_pth,very_best):


    if is_best:
        print("STIAMO SALVANDO IL VERO BEST MODEL-------------Z ",very_best)
        torch.save(state, very_best)
        wandb.save(very_best)
    else:
        torch.save(state, last_pth)
        wandb.save(last_pth)



def main(argv):

    args = parse_args(argv)
    print(args)

    

    openimages_path = "/scratch/dataset/openimages"
    kodak_path = "/scratch/dataset/kodak"
    save_path = "/scratch/Progressive/models/"


    dict_base_model = {"q2":"/scratch/base_devil/weights/q2/model.pth",
                        "q5":"/scratch/base_devil/weights/q5/model.pth"}

    
 
    wandb.init( config= args, project="ProgressiveCodec", entity="albipresta") #ddd

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
    

    epoch_enc = 0
    for epoch in range(last_epoch, args.epochs):
        print("******************************************************")
        print("epoch: ",epoch)
        start = time.time()
        #print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        num_tainable = net.print_information()
        if num_tainable > 0:
            counter = train_one_epoch( net, 
                                      criterion, 
                                      train_dataloader, 
                                      optimizer, 
                                      aux_optimizer, 
                                      epoch, 
                                      counter,
                                      list_quality= list_quality,
                                      sampling_training = args.sampling_training)

        print("finito il train della epoca")
        
        loss = valid_epoch(epoch, 
                           valid_dataloader,
                           criterion, 
                           net, pr_list = [0,10])


        lr_scheduler.step(loss)



        list_pr = [0,0.01,0.5,0.1,0.25,0.5,1,1.5,2,2.5,3,4,5,10]
        mask_pol = "point-based-std"
        bpp_t, psnr_t = test_epoch(epoch, 
                       test_dataloader,
                       net, 
                       pr_list = list_pr
                       )

        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        print("finito il test della epoca: ",bpp_t," ",psnr_t)


        end = time.time()
        print("Runtime of the epoch:  ", epoch)
        sec_to_hours(end - start) 
        print("END OF EPOCH ", epoch)


        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        if epoch%5==0 or is_best:
            net.update()
            #net.lmbda_list
            bpp, psnr,_ = compress_with_ac(net,  
                                            filelist, 
                                            device,
                                           epoch = -10, 
                                           pr_list =list_pr,  
                                            mask_pol = mask_pol,
                                            writing = None)
            
            print("COMPRESSIONE ",bpp," ",psnr)

            psnr_res = {}
            bpp_res = {}

            bpp_res["our"] = bpp
            psnr_res["our"] = psnr

            psnr_res["base"] =   [29.20, 30.59,32.26,34.15,35.91,37.72]
            bpp_res["base"] =  [0.127,0.199,0.309,0.449,0.649,0.895]



            bpp_res["tri_planet_23"] = tri_planet_23_bpp
            psnr_res["tri_planet_23"] = tri_planet_23_psnr

            bpp_res["tri_planet_22"] = tri_planet_22_bpp
            psnr_res["tri_planet_22"] = tri_planet_22_psnr

            plot_rate_distorsion(bpp_res, psnr_res,epoch_enc, eest="compression")
            
            bpp_res["our"] = bpp_t
            psnr_res["our"] = psnr_t          
            



            plot_rate_distorsion(bpp_res, psnr_res,epoch_enc, eest="model")
            epoch_enc += 1



        stringa_lambda = ""
        for lamb in args.lmbda_list:
            stringa_lambda = stringa_lambda  + "_" + str(lamb)


        name_folder = args.code + "_" + stringa_lambda + "_"
        cartella = os.path.join(save_path,name_folder) #dddd
        os.makedirs(cartella, exist_ok=True)




        last_pth, very_best =  create_savepath( cartella)
        save_checkpoint(
                {
                    "epoch": epoch,
                    "state_dict": net.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "lr_scheduler": lr_scheduler.state_dict(),
                    "aux_optimizer":aux_optimizer.state_dict() if aux_optimizer is not None else "None",
                    "args":args
      
                },
                is_best,
                last_pth,
                very_best
                )

        log_dict = {
        "train":epoch,
        "train/leaning_rate": optimizer.param_groups[0]['lr']
        #"train/beta": annealing_strategy_gaussian.bet
        }

        wandb.log(log_dict)




if __name__ == "__main__":
    main(sys.argv[1:])
