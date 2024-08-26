import torch 
import wandb
import math 
import random

from torch.nn.functional import mse_loss
from pytorch_msssim import ms_ssim
import torch.nn.functional as F 
import torchvision.transforms as transforms
from compressai.ops import compute_padding
from compressai.utils.functions import compute_msssim, compute_psnr, AverageMeter, read_image

def compute_psnr(a, b):
    mse = torch.mean((a - b)**2).item()
    return -10 * math.log10(mse)

def compute_msssim(a, b):
    return ms_ssim(a, b, data_range=1.).item()


def compute_aux_loss(aux_list):
    aux_loss_sum = 0
    for aux_loss in aux_list:
        aux_loss_sum += aux_loss
    return aux_loss_sum


def train_one_epoch(model, 
                    criterion, 
                    train_dataloader,
                      optimizer, 
                      aux_optimizer, 
                      epoch, 
                      counter,
                      sampling_training = False,
                      list_quality = None,
                      clip_max_norm = 1.0,
                      video = False):
    model.train()
    device = next(model.parameters()).device


    loss = AverageMeter()
    bpp_loss = AverageMeter()
    mse_loss = AverageMeter()
    bpp_scalable = AverageMeter()
    bpp_main = AverageMeter()
    kd_enh = AverageMeter()
    kd_base = AverageMeter()

    mutual_info = AverageMeter()




    lmbda_list = model.lmbda_list
    for i, d in enumerate(train_dataloader):
        d = d.to(device)

        optimizer.zero_grad()
        if aux_optimizer is not None:
            aux_optimizer.zero_grad()

        if sampling_training:
            quality_index =  random.randint(0, len(list_quality) - 1)
            quality = list_quality[quality_index]
            out_net = model.forward_single_quality(d, quality = quality, training = False)
            out_criterion = criterion(out_net, d)
        else:
            out_net = model(d, quality = list_quality)
            out_criterion = criterion(out_net, d)

        out_criterion["loss"].backward()
        if aux_optimizer is not None:
            aux_loss = model.aux_loss() if video is False else compute_aux_loss(model.aux_loss())
            aux_loss.backward()
            aux_optimizer.step()


        if clip_max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_max_norm)
        optimizer.step()


        loss.update(out_criterion["loss"].clone().detach())
        mse_loss.update(out_criterion["mse_loss"].mean().clone().detach())
        bpp_loss.update(out_criterion["bpp_loss"].clone().detach())
        if "bpp_scalable" in list(out_criterion.keys()):
            bpp_scalable.update(out_criterion["bpp_scalable"].clone().detach())

        #bpp_main.update(out_criterion["bpp_base"].clone().detach())

        if "kd_enh" in list(out_criterion.keys()):
            kd_enh.update(out_criterion["kd_enh"].clone().detach())
        if "kd_base" in list(out_criterion.keys()):
            kd_enh.update(out_criterion["kd_base"].clone().detach())
        if "mutual" in list(out_criterion.keys()):
            mutual_info.update(out_criterion["mutual"])


        wand_dict = {
            "train_batch": counter,
            "train_batch/loss": out_criterion["loss"].clone().detach().item(),
            "train_batch/bpp_total": out_criterion["bpp_loss"].clone().detach().item(),
            "train_batch/mse":out_criterion["mse_loss"].mean().clone().detach().item(),
        }
        wandb.log(wand_dict)

        if "bpp_scalable" in list(out_criterion.keys()):
            wand_dict = {
                "train_batch": counter,
                "train_batch/bpp_progressive":out_criterion["bpp_scalable"].clone().detach().item(),
            }
            wandb.log(wand_dict)


        if "kd_enh" in list(out_criterion.keys()):
            wand_dict = {
                "train_batch": counter, 
                "train_batch/kd_enh": out_criterion["kd_enh"].clone().detach().item(),
            }
            wandb.log(wand_dict)
        if "kd_base" in list(out_criterion.keys()):
            wand_dict = {
                "train_batch": counter, 
                "train_batch/kd_base": out_criterion["kd_base"].clone().detach().item(),
            }
            wandb.log(wand_dict)
        if "mutual" in list(out_criterion.keys()):
            wand_dict = {
                "train_batch": counter, 
                "train_batch/mutual": out_criterion["mutual"],
            }
            wandb.log(wand_dict)
        counter += 1





        if i % 1000 == 0:
            print(
                f"Train epoch {epoch}: ["
                f"{i*len(d)}/{len(train_dataloader.dataset)}"
                f" ({100. * i / len(train_dataloader):.0f}%)]"
                f'\tLoss: {out_criterion["loss"].item():.3f} |'
                f'\tMSE loss: {out_criterion["mse_loss"].mean().item() * 255 ** 2 / 3:.3f} |'
                f'\tBpp loss: {out_criterion["bpp_loss"].item():.2f} |'
               f"\tAux loss: {0.000:.2f}"
            )
    
    log_dict = {
        "train":epoch,
        "train/loss": loss.avg,
        "train/bpp": bpp_loss.avg,
        "train/mse": mse_loss.avg,
        "train/bpp_progressive":bpp_scalable.avg,
        }
    wandb.log(log_dict)

    if "bpp_scalable" in list(out_criterion.keys()):
        log_dict = {
            "train":epoch,
            "train/bpp_progressive":bpp_scalable.avg,
            }
        wandb.log(log_dict) 

    if "mutual" in list(out_criterion.keys()):
        log_dict = {
            "train":epoch,
            "train/mutual":mutual_info.avg,
            }
        wandb.log(log_dict) 


    return counter



    


def valid_epoch(epoch, test_dataloader,criterion, model, pr_list = [0.05]):
    #pr_list =  [0] +  pr_list  + [-1]
    model.eval()
    device = next(model.parameters()).device

    loss = AverageMeter() 
    bpp_loss =AverageMeter() 
    mse_lss = AverageMeter() 
    
    psnr = AverageMeter() 
    mutual_info = AverageMeter()

    with torch.no_grad():
        for d in test_dataloader:

            d = d.to(device)
            for _,p in enumerate(pr_list):

                out_net = model.forward_single_quality(d, quality = p, training = False, mask_pol = "point-based-std")

                psnr_im = compute_psnr(d, out_net["x_hat"])
                batch_size_images, _, H, W =d.size()
                num_pixels = batch_size_images * H * W
                denominator = -math.log(2) * num_pixels
                likelihoods = out_net["likelihoods"]
                bpp = (torch.log(likelihoods["y"]).sum() + torch.log(likelihoods["z"]).sum())/denominator

                    
                bpp_loss.update(bpp)

                mse_lss.update(mse_loss(d, out_net["x_hat"]))
                psnr.update(psnr_im) 

                        #out_net = model(d)
                out_criterion = criterion(out_net, d) #dddddd

                loss.update(out_criterion["loss"].clone().detach())

         

                    
    log_dict = {
            "valid":epoch,
            "valid/loss": loss.avg,
            "valid/bpp":bpp_loss.avg,
            "valid/mse": mse_lss.avg,
            "valid/psnr":psnr.avg,

         #   "test/y_loss_"+ name[i]: y_loss[i].avg,
            }
    wandb.log(log_dict)
    if "mutual" in list(out_criterion.keys()):
        log_dict = {
            "valid":epoch,
            "valid/mutual":mutual_info.avg,

            }

    wandb.log(log_dict)
    return loss.avg


def test_epoch(epoch, test_dataloader, model, pr_list):
    model.eval()
    device = next(model.parameters()).device


    bpp_loss =[AverageMeter()  for _ in range(len(pr_list))] 
    psnr = [AverageMeter()  for _ in range(len(pr_list))]
    mutual_info = [AverageMeter()  for _ in range(len(pr_list))]

    with torch.no_grad():
        for d,_ in test_dataloader:
            d = d.to(device)
            for j,p in enumerate(pr_list):
                out_net = model.forward_single_quality(d, quality = p, training = False,  mask_pol = "point-based-std")

                psnr_im = compute_psnr(d, out_net["x_hat"])
                batch_size_images, _, H, W =d.size()
                num_pixels = batch_size_images * H * W
                denominator = -math.log(2) * num_pixels #dddd
                likelihoods = out_net["likelihoods"]
                bpp = (torch.log(likelihoods["y"]).sum() + torch.log(likelihoods["z"]).sum())/denominator


                psnr[j].update(psnr_im)
                bpp_loss[j].update(bpp)



    for i in range(len(pr_list)):
        if i== 0:
            name = "test_base"
        elif i == len(pr_list) - 1:
            name = "test_complete"
        else:
            c = str(pr_list[i])
            name = "test_quality_" + c 
        
        log_dict = {
            name:epoch,
            name + "/bpp":bpp_loss[i].avg,
            name + "/psnr":psnr[i].avg,
            }

        wandb.log(log_dict)

        if "mutual" in list(out_net):
            log_dict = {
                name:epoch,
                name + "/mutual":mutual_info[i].avg
                }
            wandb.log(log_dict)

    return [bpp_loss[i].avg for i in range(len(bpp_loss))], [psnr[i].avg for i in range(len(psnr))]

import time
def sec_to_hours(seconds):
    a=str(seconds//3600)
    b=str((seconds%3600)//60)
    c=str((seconds%3600)%60)
    d=["{} hours {} mins {} seconds".format(a, b, c)]
    print(d[0])

def compress_with_ac(model,  
                     filelist, 
                     device, 
                     epoch, 
                     pr_list = [0.05,0.01], 
                     mask_pol = None, 
                       writing = None, 
                       base = False,
                       
                       cheating = False,
                       customs_maps =False,
                       save_images = False):

    l = len(pr_list)
    print("ho finito l'update")
    bpp_loss = [AverageMeter() for _ in range(l)]
    psnr =[AverageMeter() for _ in range(l)]
    mssim = [AverageMeter() for _ in range(l)]
    dec_time = [AverageMeter() for _ in range(l)]
    only_dec_time = [AverageMeter() for _ in range(l)]


    for i,d in enumerate(filelist):
        print("l'immagine d: ",d," ",i)



        if customs_maps is True:
            for p in model.parameters():
                p.requires_grad = True
            #model.print_information()
            custom_map =  extract_dec_importance_map(d,model)

            for p in model.parameters():
                p.requires_grad = False
        else:
             custom_map = None 

        with torch.no_grad():
            x = read_image(d).to(device)
            nome_immagine = d.split("/")[-1].split(".")[0]
            x = x.unsqueeze(0) 
            h, w = x.size(2), x.size(3)
            pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
            x_padded = F.pad(x, pad, mode="constant", value=0)
        

            for j,p in enumerate(pr_list):
                
                lev = "level_" + str(j)

                

                data =  model.compress(x_padded, quality =p, 
                                       mask_pol = mask_pol, cust_map = custom_map if p > 0 else None)
                #if j%8==0:
                #    print(sum(len(s[0]) for s in data["strings"][0]))
                start = time.time()
                out_dec = model.decompress(data["strings"], data["shape"], 
                                                quality = p,
                                                  mask_pol = mask_pol, 
                                                  cust_map = custom_map if p > 0 else None,
                                                )
                end = time.time()
                #print("Runtime of the epoch:  ", epoch)
                decoded_time = end-start
                #sec_to_hours(end - start) 
                out_dec["x_hat"] = F.pad(out_dec["x_hat"], unpad)
                out_dec["x_hat"].clamp_(0.,1.)   

                if save_images:
                    output_image = transforms.ToPILImage()(out_dec["x_hat"].squeeze(0))  
                    output_image.save("/scratch/images_k/" + nome_immagine + str(j) + ".png" )

                psnr_im = compute_psnr(x, out_dec["x_hat"])
                ms_ssim_im = compute_msssim(x, out_dec["x_hat"])
                ms_ssim_im = -10*math.log10(1 - ms_ssim_im )
                psnr[j].update(psnr_im)
                mssim[j].update(ms_ssim_im)
                dec_time[j].update(decoded_time)
                #only_dec_time[j].update(out_dec["time"])
            
                size = out_dec['x_hat'].size()
                num_pixels = size[0] * size[2] * size[3]

                # calcolo lo stream del base 
                if base is True:
                    bpp = sum(len(s[0]) for s in data["strings"]) * 8.0 / num_pixels

                else:
                    data_string_scale = data["strings"][0] # questo è una lista
                    bpp_scale = sum(len(s[0]) for s in data_string_scale) * 8.0 / num_pixels #ddddddd
                        
                    data_string_hype = data["strings"][1]
                    bpp_hype = sum(len(s) for s in data_string_hype) * 8.0 / num_pixels
                    bpp = bpp_hype + bpp_scale if cheating is False or j == 0 else bpp_scale


                bpp_loss[j].update(bpp)

                if writing is not None:
                    fls = writing + "/"  +  lev + "_.txt"
                    f=open(fls , "a+")
                    f.write("SEQUENCE "  +   nome_immagine + " BITS " +  str(bpp) + " PSNR " +  str(psnr_im)  + " MSSIM " +  str(ms_ssim_im) + "\n")
                    f.close()  

    if epoch > -1:
        for i in range(len(pr_list)):
            if i== 0:
                name = "compress_base"
            elif i == len(pr_list) - 1:
                name = "compress_complete"
            else:
                c = str(pr_list[i])
                name = "compress_quality_" + c
            
            log_dict = {
                name:epoch,
                name + "/bpp":bpp_loss[i].avg,
                name + "/psnr":psnr[i].avg,
                }

            wandb.log(log_dict)


    
    #print("enhanced compression results : ",bpp_loss.avg," ",psnr_val.avg," ",mssim_val.avg)
    if writing is not None:
        for j,p in enumerate(pr_list):
            lev =  "level_" + str(j)
            fls = writing +  "/" +  lev + "_.txt"
            f=open(fls , "a+")
            f.write("SEQUENCE "  +   "AVG " + "BITS " +  str(bpp_loss[j].avg) + " YPSNR " +  str(psnr[j].avg)  + " YMSSIM " +  str(mssim[j].avg) + "\n")
            f.close()
    return [bpp_loss[i].avg for i in range(len(bpp_loss))], [psnr[i].avg for i in range(len(psnr))], [dec_time[i].avg for i in range(len(dec_time))] #, [only_dec_time[i].avg for i in range(len(only_dec_time))]

########################################################################################################
########################################################################################################
########################################################################################################

import torch
import torch.nn.functional as F
from torch.autograd import Variable



def custom_mse_function(output, target):
    # Esempio di funzione di loss: quadrato della differenza tra il valore del pixel e un valore target
    
    loss = (255**2)*torch.mean((output - target)**2)
    return loss


def get_scale_table(min=0.11, max=256, levels=64):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))



def gaussian_sampling_torch(mean, std):
    # Genera campioni casuali da una distribuzione gaussiana
    samples = torch.normal(mean, std)
    return samples

def build_indexes( scales,scale_table):
    scales = torch.relu(scales)
    indexes = scales.new_full(scales.size(), len(scale_table) - 1).int()
    for s in scale_table[:-1]:
        indexes -= (scales <= s).int()

    

    return indexes


def extract_dec_importance_map(f,net, perc = 0.001,device = "cuda"):
    x = read_image(f).to(device)

    x = x.unsqueeze(0) 
   
    h, w = x.size(2), x.size(3)
    pad, unpad = compute_padding(h, w, min_div=2**6)  # pad to allow 6 strides of 2
    x_padded = F.pad(x, pad, mode="constant", value=0)

    
    out_dec = net.forward_single_quality(x_padded, 
                                         quality = 0, 
                                         training = False, 
                                         mask_pol = "point-based-std")
    out_dec_completo = net.forward_single_quality(x_padded, 
                                                  quality = 0.000001,
                                                  training = False, 
                                                  mask_pol = "point-based-std")

    
    mu_b = out_dec["mu"]
    std_b = out_dec["std"]
    mu_e = out_dec_completo["mu"]
    std_e = out_dec_completo["std"]

    print("torch min: ",torch.min(std_b)," ",torch.min(std_e))

    mu_progressive = mu_b + mu_e
    scale_table = get_scale_table().to(std_b.device)
    std_b_index = build_indexes(std_b,scale_table).to(std_b.device)
    std_e_index = build_indexes(std_e,scale_table).to(std_e.device)

    std_b = scale_table[std_b_index]
    std_e = scale_table[std_e_index]

    

    
    y_hat_b_sampled = gaussian_sampling_torch(mu_b,std_b)
    y_hat_enh_sampled = gaussian_sampling_torch(mu_e,std_e) + y_hat_b_sampled



    difference_input = torch.abs(y_hat_b_sampled - y_hat_enh_sampled) #ffffFFF


    target = net.base_net.g_s[1](y_hat_enh_sampled).detach().cpu()

    target = target.to(device)


    y_hat_b_sampled = Variable(y_hat_b_sampled, requires_grad=True)
    x_hat_b = net.base_net.g_s[1](y_hat_b_sampled)
    loss = (255**2)*torch.mean((x_hat_b - target)**2)

    net.base_net.g_s[1].zero_grad()  
    loss.backward(retain_graph=True)

    # Ora image.grad contiene il gradiente della loss rispetto a ciascun pixel dell'immagine
    gradient = y_hat_b_sampled.grad
    gradient = gradient.to(device)
    gradient = torch.abs(gradient)*difference_input #dddd

    #print("l'importance map ha questa dimensione: ",gradient.shape)

    map = gradient #* difference_input
    #print("map shape: ",map.shape)
    #print("massimo e minimo: ",torch.max(map),"  ",torch.min(map))
    #print("uniquens: ",torch.unique(map))

    return map


