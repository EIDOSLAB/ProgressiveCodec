
import os
from pathlib import Path
from torch.autograd import Variable

from PIL import Image
from torch.utils.data import Dataset
import torch


class ImageFolder(Dataset):


    def __init__(self, root, num_images = 24000, transform=None, split="train", names = False):
        splitdir = Path(root) / split / "data"
        self.names = names
        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples =[]# [f for f in splitdir.iterdir() if f.is_file()]

        num_images = num_images
            
        print("entro qui per il dataset")
        for i,f in enumerate(splitdir.iterdir()):
            if i%50000==0:
                print(i)
            if i <= num_images: 
                if f.is_file() and i < num_images:
                    self.samples.append(f)
            else:
                break
        print("lunghezza: ",len(self.samples))
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB") #dd
        #st = str(self.samples[index])
        #nome = st.split("/")[-1].split(".")[0]
        
        return self.transform(img)



    def __len__(self):
        return len(self.samples)
    



class TestKodakDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.data_dir = data_dir
        self.transform = transform
        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = [os.path.join(self.data_dir,f) for f in os.listdir(self.data_dir)]

    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        #transform = transforms.Compose([transforms.CenterCrop(256), transforms.ToTensor()])
        
        return self.transform(image), image_ori

    def __len__(self):
        return len(self.image_path)
    





class MaskImageFolder(Dataset):


    def __init__(self, 
                 root,
                 net, 
                 num_images = 24000, 
                 transform=None, 
                 split="train", 
                 pr = 0.001, 
                 device = "cuda",
                 normalize = False,
                 post_optimized =True):
        splitdir = Path(root) / split / "data"
        self.pr = pr

        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples =[]# [f for f in splitdir.iterdir() if f.is_file()]

        self.normalize = normalize
        num_images = num_images
            
        print("entro qui per il dataset")
        for i,f in enumerate(splitdir.iterdir()):
            if i%50000==0:
                print(i)
            if i <= num_images: 
                if f.is_file() and i < num_images:
                    self.samples.append(f)
            else:
                break

        self.transform = transform
        net.update()
        if post_optimized:
            self.net = net.base_net 
        else:
            self.net = net 

        
        # Freezare tutti i parametri tranne quelli di `self.g_s`
        for name, param in self.net.named_parameters():
            if "g_s" not in name:
                param.requires_grad = False

        assert self.net.multiple_encoder 
        assert self.net.multiple_decoder

        self.device = device

        self.net.print_information()
        


    
    def extract_latentspace_and_rec(self,x):
        x = x.unsqueeze(0)
        x = x.to(self.device)

        #print("la x che dimensione ha: ",x.shape)
        out_base = self.net.forward_single_quality(x, 
                                                  quality = 0, 
                                                  training = False, 
                                                  force_enhanced = False,
                                                  mask_pol = "point-based-std")
        
        out_p0 = self.net.forward_single_quality(x, 
                                                  quality = 0.0001, 
                                                  training = False, 
                                                  force_enhanced = True,
                                                  mask_pol = "point-based-std")
        out_dec_completo = self.net.forward_single_quality(x, 
                                                  quality = 10,
                                                  training = False,
                                                  force_enhanced = True, 
                                                  mask_pol = "point-based-std")
        
        
        y_hat_b = out_base["y_hat"]
        mu_b = out_base["mu"]
        std_b = out_base["std"]

        y_hat_p0 = out_p0["y_hat"]
        y_hat_p0 = Variable(y_hat_p0, requires_grad=True)
        mu_p0 = out_p0["mu"]
        std_p0 = out_p0["std"]
        x_hat_p0 = self.net.g_s[1](y_hat_p0)

        y_hat_e = out_dec_completo["y_hat"]
        target = self.net.g_s[1](y_hat_e)



        loss = (255**2)*torch.mean((x_hat_p0 - target)**2) #dddd

        self.net.g_s[1].zero_grad()  
        loss.backward()


        gradient = y_hat_p0.grad
        gradient = gradient.to(self.device)
        difference_input = torch.abs(y_hat_p0 - y_hat_e)
        map = torch.abs(gradient)*difference_input

        if self.normalize:
            map = torch.sigmoid(map)

        params_b = torch.cat([mu_b,std_b],dim = 1)
        params_p0 = torch.cat([mu_p0,std_p0],dim = 1)


        y_hat_b = y_hat_b.squeeze(0).detach()#.cpu()
        y_hat_p0 = y_hat_p0.squeeze(0).detach()#.cpu()
        
        params_b = params_b.squeeze(0).detach()#.cpu()
        params_p0 = params_p0.squeeze(0).detach()#.cpu()#dddd
        map = map.squeeze(0).detach()




        #min_val = torch.min(map)
        #max_val = torch.max(map)
        #norm_map = (map - min_val) / (max_val - min_val)

        return y_hat_b,y_hat_p0,params_b,params_p0, map



    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            img = self.transform(img) 

        y_hat_b,y_hat_p0,params_b,params_p0, map = self.extract_latentspace_and_rec(img) #dddd
        return img, y_hat_b,y_hat_p0,params_b,params_p0, map



    def __len__(self):
        return len(self.samples)
    




class MaskTestKodakDataset(Dataset):
    def __init__(self, data_dir, net, transform, post_optimized = True, device = "cuda",normalize=False):
        self.data_dir = data_dir
        self.transform = transform
        self.pr = 0.00000

        if not os.path.exists(data_dir):
            raise Exception(f"[!] {self.data_dir} not exitd")
        self.image_path = [os.path.join(self.data_dir,f) for f in os.listdir(self.data_dir)]

        self.normalize = normalize
        net.update()
        if post_optimized:
            self.net = net.base_net 
        else:
            self.net = net 

        
        # Freezare tutti i parametri tranne quelli di `self.g_s`
        for name, param in self.net.named_parameters():
            if "g_s" not in name:
                param.requires_grad = False

        assert self.net.multiple_encoder 
        assert self.net.multiple_decoder

        self.device = device

        self.net.print_information()

    def extract_latentspace_and_rec(self,x):
        x = x.unsqueeze(0)
        x = x.to(self.device)

        
        out_base = self.net.forward_single_quality(x, 
                                                  quality = 0, 
                                                  training = False, 
                                                  force_enhanced = False,
                                                  mask_pol = "point-based-std")
        
        out_p0 = self.net.forward_single_quality(x, 
                                                  quality = 0.001, 
                                                  training = False, 
                                                  force_enhanced = True,
                                                  mask_pol = "point-based-std")
        out_dec_completo = self.net.forward_single_quality(x, 
                                                  quality = 10,
                                                  training = False,
                                                  force_enhanced = True, 
                                                  mask_pol = "point-based-std")
        
        
        y_hat_b = out_base["y_hat"]
        mu_b = out_base["mu"]
        std_b = out_base["std"]

        y_hat_p0 = out_p0["y_hat"]
        y_hat_p0 = Variable(y_hat_p0, requires_grad=True)
        mu_p0 = out_p0["mu"]
        std_p0 = out_p0["std"]
        x_hat_p0 = self.net.g_s[1](y_hat_p0)

        y_hat_e = out_dec_completo["y_hat"]
        mu_e = out_dec_completo["mu"]
        std_e = out_dec_completo["std"]
        target = self.net.g_s[1](y_hat_e)



        loss = (255**2)*torch.mean((x_hat_p0 - target)**2) #dddd

        self.net.g_s[1].zero_grad()  
        loss.backward()


        gradient = y_hat_p0.grad
        gradient = gradient.to(self.device)
        difference_input = torch.abs(y_hat_p0 - y_hat_e)
        map = torch.abs(gradient)*difference_input

        if self.normalize:
            map = torch.sigmoid(map)


        params_b = torch.cat([mu_b,std_b],dim = 1)
        params_p0 = torch.cat([mu_p0,std_p0],dim = 1)


        y_hat_b = y_hat_b.squeeze(0).detach()#.cpu()
        y_hat_p0 = y_hat_p0.squeeze(0).detach()#.cpu()
        
        params_b = params_b.squeeze(0).detach()#.cpu()
        params_p0 = params_p0.squeeze(0).detach()#.cpu()#dddd
        map = map.squeeze(0).detach()

        

        return y_hat_b,y_hat_p0,params_b,params_p0, map



    def __getitem__(self, item):
        image_ori = self.image_path[item]
        image = Image.open(image_ori).convert('RGB')
        image = self.transform(image)
        y_hat_b,y_hat_p0,params_b,params_p0, map = self.extract_latentspace_and_rec(image)
        return image, y_hat_b,y_hat_p0,params_b,params_p0, map

    def __len__(self):
        return len(self.image_path)
    