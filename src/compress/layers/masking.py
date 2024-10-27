import torch
import torch.nn as nn
from compress.ops import LowerBound, ste_round
from compress.layers import conv3x3, subpel_conv3x3




class ChannelMask(nn.Module):

    def __init__(self, 
                 mask_policy, 
                 scalable_levels,
                 dim_chunk, 
                 num_levels, 
                 gamma_bound = 1e-9, 
                 double_dim = False):
        super().__init__()

        self.mask_policy = mask_policy 
        self.scalable_levels = scalable_levels 
        self.quality_list = [i for i in range(self.scalable_levels)]
        self.dim_chunk =dim_chunk 
        self.num_levels = num_levels
        self.double_dim = double_dim
        
        if self.double_dim:
            print("vado qua dove il double dim è raddoppiato!!!")
            self.input_dim = self.dim_chunk*2
        else: 
            self.input_dim = self.dim_chunk


        if self.mask_policy == "learnable-mask-quantile":
            self.mask_conv = nn.ModuleList(nn.Sequential(
                            conv3x3(self.input_dim,self.dim_chunk),
                            nn.ReLU(),
                            conv3x3(self.input_dim,self.dim_chunk,  stride=2),
                            nn.ReLU(),
                            subpel_conv3x3(self.input_dim,self.dim_chunk, 2),
                            nn.ReLU(),
                            conv3x3(self.input_dim,self.dim_chunk),
                            nn.ReLU()                           
                            ) for _ in range(self.num_levels)
            )

        elif self.mask_policy == "single-learnable-mask-quantile":
            self.mask_conv = nn.Sequential(
                            conv3x3(self.input_dim,self.dim_chunk),
                            nn.ReLU(),
                            conv3x3(self.input_dim,self.dim_chunk,  stride=2),
                            nn.ReLU(),
                            subpel_conv3x3(self.input_dim,self.dim_chunk, 2),
                            nn.ReLU(),
                            conv3x3(self.input_dim,self.dim_chunk),
                            nn.Sigmoid()                           
                            ) 
            
        elif self.mask_policy == "single-learnable-mask-gamma":
            self.gamma = [
                        torch.nn.Parameter(torch.ones((self.scalable_levels - 2, self.dim_chunk))) 
                        for _ in range(self.num_levels)
            ]
            self.mask_conv = nn.Sequential(
                            conv3x3(self.input_dim,self.dim_chunk),
                            nn.ReLU(),
                            conv3x3(self.input_dim,self.dim_chunk,  stride=2),
                            nn.ReLU(),
                            subpel_conv3x3(self.input_dim,self.dim_chunk, 2),
                            nn.ReLU(),
                            conv3x3(self.input_dim,self.dim_chunk),
                            nn.Sigmoid()                           
                            ) 
            
            self.gamma_lower_bound = LowerBound(gamma_bound)


        elif self.mask_policy == "learnable-mask-gamma":
            self.gamma = [
                        torch.nn.Parameter(torch.ones((self.scalable_levels - 2, self.dim_chunk))) 
                        for _ in range(self.num_levels)
            ]
            self.mask_conv = nn.ModuleList(
                    nn.Sequential(torch.nn.Conv2d(in_channels=self.input_dim, out_channels=self.dim_chunk, kernel_size=1, stride=1),) for _ in range(self.num_levels)
                    )
            
            self.gamma_lower_bound = LowerBound(gamma_bound)
        
        
        elif self.mask_policy == "learnable-mask-nested":
            self.mask_conv = nn.ModuleList(
                                nn.ModuleList(
                                    nn.Sequential(torch.nn.Conv2d(in_channels=self.input_dim, 
                                    out_channels=self.dim_chunk, 
                                    kernel_size=1, 
                                    stride=1),) for _ in range(self.scalable_levels -2)
                                    ) for _ in range(self.num_levels) )
        elif self.mask_policy == "single-learnable-mask-nested":
            self.mask_conv = nn.Sequential(
                            conv3x3(self.input_dim,self.dim_chunk),
                            nn.ReLU(),
                            conv3x3(self.input_dim,self.dim_chunk,  stride=2),
                            nn.ReLU(),
                            subpel_conv3x3(self.input_dim,self.dim_chunk, 2),
                            nn.ReLU(),
                            conv3x3(self.input_dim,self.dim_chunk),
                            nn.Sigmoid()                           
                            ) 
        
        elif self.mask_policy == "three-levels-learnable":
            print("STO ENTRANDO QUA,DEVO COSTRUIRLA BENE!!!")
            self.mask_conv = nn.ModuleList(nn.Sequential(
                            conv3x3(self.input_dim,self.input_dim),
                            nn.ReLU(),
                            conv3x3(self.input_dim,self.input_dim,  stride=2),
                            nn.ReLU(),
                            subpel_conv3x3(self.input_dim,self.dim_chunk, 2),
                            nn.ReLU(),
                            conv3x3(self.dim_chunk,self.dim_chunk),
                            nn.Sigmoid()                           
                            ) for _ in range(self.num_levels) )

            #print(self.mask_conv)          

    def apply_noise(self, mask, tr):
            if tr:
                mask = ste_round(mask)
                #mask = mask + (torch.rand_like(mask) - 0.5)
                #mask = mask + mask.round().detach() - mask.detach()  # Differentiable torch.round()   
                
            else:
                mask = torch.round(mask)
            return mask
    

    def delta_mask(self, scale,  pr_bar, pr):
        shapes = scale.shape
        bs, ch, w,h = shapes
        assert pr_bar <= pr
        if pr >= 10:
            return torch.ones_like(scale).to(scale.device)
        elif pr == 0:
            return torch.zeros_like(scale).to(scale.device) 
        assert scale is not None 
        pr = 10 if pr > 10 else pr
        pr = pr*0.1
        pr = 1.0 - pr

        pr_bar = 10 if pr_bar > 10 else pr_bar
        pr_bar = pr_bar*0.1
        pr_bar = 1.0 - pr_bar
        res = torch.zeros_like(scale).to(scale.device)   
        for j in range(bs):
            scale_b = scale[j,:,:,:]
            scale_b = scale_b.ravel()
            quantile_bar = torch.quantile(scale_b, pr_bar)
            quantile = torch.quantile(scale_b, pr)
            res_b = quantile_bar >= scale_b >= quantile #if "inverse" not in mask_pol else  scale_b <= quantile
            res_b = res_b.reshape(ch,w,h)
            res_b = res_b.to(scale.device)
            res[j] = res_b             

    def forward(self,
                scale,  
                scale_base = None, 
                slice_index = 0,  
                pr = 0, 
                mask_pol = "point-based-std",
                cust_map = None):

        if cust_map is not None:
            #cust_map = cust_map.unsqueeze(0).to(cust_map.device)
            shapes = cust_map.shape
            bs, ch, w,h = shapes
            if pr >= 10:
                return torch.ones_like(cust_map).to(scale.device)
            elif pr == 0:
                return torch.zeros_like(cust_map).to(scale.device)           
            pr = 10 if pr > 10 else pr
            pr = pr*0.1
            pr_bis = 1.0 - pr
            res = torch.zeros_like(cust_map).to(scale.device)            

            for j in range(bs):
                cust_map_b = cust_map[j,:,:,:]
                cust_map_b = cust_map_b.ravel()
                quantile = torch.quantile(cust_map_b, pr_bis)
                res_b = cust_map_b >= quantile #if "inverse" not in mask_pol else  scale_b <= quantile
                res_b = res_b.reshape(ch,w,h)
                res_b = res_b.to(scale.device)
                res[j] = res_b

            #print("struttura della maschera for: ",pr,": ", torch.unique(res,return_counts = True))
            return res.to(scale.device) 
        
        if mask_pol is None:
            mask_pol = self.mask_policy

        shapes = scale.shape
        bs, ch, w,h = shapes

        if mask_pol is None: 
            return torch.ones_like(scale).to(scale.device)
        
        if mask_pol == "point-based-std":
            if pr >= 10:
                return torch.ones_like(scale).to(scale.device)
            elif pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            assert scale is not None 
            pr = 10 if pr > 10 else pr
            pr = pr*0.1
            pr_bis = 1.0 - pr
            res = torch.zeros_like(scale).to(scale.device)
            for j in range(bs):
                scale_b = scale[j,:,:,:]
                scale_b = scale_b.ravel()
                quantile = torch.quantile(scale_b, pr_bis)
                res_b = scale_b >= quantile #if "inverse" not in mask_pol else  scale_b <= quantile
                res_b = res_b.reshape(ch,w,h)
                res_b = res_b.to(scale.device)
                res[j] = res_b
            return res.float().reshape(bs,ch,w,h).to(torch.float).to(scale.device)



        elif mask_pol == "two-levels":
            return torch.zeros_like(scale).to(scale.device) if pr == 0 else torch.ones_like(scale).to(scale.device)
        elif mask_pol == "three-levels-std":
            if pr == 0: 
                return torch.zeros_like(scale).to(scale.device)
            elif pr == 2:
                return torch.ones_like(scale).to(scale.device)
            else:
                assert scale is not None 

                pr = 0.8
                res = torch.zeros_like(scale).to(scale.device)
                for j in range(bs):
                    scale_b = scale[j,:,:,:]
                    scale_b = scale_b.ravel()
                    quantile = torch.quantile(scale_b, pr)
                    res_b = scale_b >= quantile #if "inverse" not in mask_pol else  scale_b <= quantile
                    res_b = res_b.reshape(ch,w,h)
                    res_b = res_b.to(scale.device)
                    res[j] = res_b
                return res.float().reshape(bs,ch,w,h).to(torch.float).to(scale.device)
        elif mask_pol == "three-levels-learnable":
            if pr == 0: 
                return torch.zeros_like(scale).to(scale.device)
            elif pr == 2:
                return torch.ones_like(scale).to(scale.device)
            else:
                #assert scale_base is not None
                if self.double_dim:
                    importance_map = self.mask_conv[slice_index](scale_base)
                else:
                    importance_map =  self.mask_conv[slice_index](scale) 
                return ste_round(importance_map)
                           
        elif mask_pol == "random":
            shape = scale.shape
            pr = pr*10
            total_elements = torch.prod(torch.tensor(shape)).item()
            num_ones = int(total_elements * (pr / 100.0))
            tensor = torch.zeros(shape).to(scale.device)
            indices = torch.randperm(total_elements)[:num_ones]
            # Impostiamo gli elementi selezionati su 1
            tensor.view(-1)[indices] = 1  

            #print("percentage random: ",(total_elements - num_ones)/total_elements)
            return tensor   
        elif mask_pol == "scalable_res":
            if pr == 0:
                return torch.zeros_like(scale).to(scale.device)
            elif pr ==10:
                return torch.ones_like(scale).to(scale.device)
            else:
                pr = pr*0.1
                ones_channel = int(320*pr) 
                canale_inizio = slice_index*32
                canale_fine = 32*(slice_index + 1)
                if ones_channel >= canale_fine: 
                    c = torch.ones_like(scale).to(scale.device)
                    return c 
                elif ones_channel < canale_inizio:
                    c = torch.zeros_like(scale).to(scale.device)
                    return c 
                else: 
                    c = torch.zeros_like(scale).to(scale.device)
                    remainder = ones_channel%32
                    c[:,remainder:,:,:] = 1 
                    return c.to(scale.device)

        else:
            raise NotImplementedError()