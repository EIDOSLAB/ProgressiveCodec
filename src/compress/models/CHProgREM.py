import torch
import torch.nn as nn
from compress.ops import ste_round
from .CHProg_cnn import ChannelProgresssiveWACNN
from .utils import ResidualBlock, ResidualBlockSmall
from torch.nn import Sequential as Seq
import time




class LatentRateReduction(nn.Module):


        def __init__(self, dim_chunk = 32, mu_std = False, dimension = "middle" ):
            super().__init__()
            self.dim_block = dim_chunk

            self.mu_std = mu_std
            N = self.dim_block



            if dimension != "big": #dd
                self.enc_base_entropy_params = Seq(
                    ResidualBlock(2 * N ,  N),
                    ResidualBlock( N, N),
                    )
                
                self.enc_enh_entropy_params = Seq(
                    ResidualBlock(2 * N if self.mu_std else N ,  N),
                    ResidualBlock( N, N),
                    )

                self.enc_base_rep = Seq(
                    ResidualBlock(N, N),
                    ResidualBlock(N, N),
                    )

                self.enc = Seq(
                    ResidualBlock(3 * N, 2 * N),
                    ResidualBlock(2 * N, 2 * N),
                    ResidualBlock(2 * N, 2 * N if self.mu_std else N),
                )
            else:
                self.enc_base_entropy_params = Seq(
                    ResidualBlock(2 * N ,  N),
                    ResidualBlock( N, N),
                    ResidualBlock( N, N),
                    )
                
                self.enc_enh_entropy_params = Seq(
                    ResidualBlock(2 * N if self.mu_std else N ,  N),
                    ResidualBlock( N, N),
                    ResidualBlock( N, N),
                    )            

                self.enc_base_rep = Seq(
                    ResidualBlock(N, N),
                    ResidualBlock(N, N),
                    ResidualBlock(N, N),
                    )   

                self.enc = Seq(
                    ResidualBlock(3 * N, 2 * N),
                    ResidualBlock(2 * N, 2 * N),
                    ResidualBlock(2 * N, 2 * N),
                    ResidualBlock(2 * N, 2 * N if self.mu_std else N),
                )           



        def forward(self, x_base, entropy_params_base, entropy_params_prog, att_mask):

            identity = entropy_params_prog

            f_ent_prog = self.enc_enh_entropy_params(entropy_params_prog) 
            f_latent = self.enc_base_rep(x_base)
            f_ent_base = self.enc_base_entropy_params(entropy_params_base)            

            ret = self.enc(torch.cat([f_latent, f_ent_base, f_ent_prog], dim=1)) #fff
            ret = ret*att_mask
            #ret = self.enc(torch.cat([f_latent, f_ent_prog], dim=1)) 
            ret += identity
            return ret

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################


class MaskEstractor(nn.Module):


    def __init__(self, N = 320, portion = "big", normalize = False):
        super().__init__()
        self.N = N
        self.portion = portion
        self.normalize = normalize




        self.enc_base_entropy_params = Seq( ResidualBlock(2 * N,  N),ResidualBlock( N,  N) )
            
        self.enc_p0_entropy_params = Seq(
                 ResidualBlock(2 *N,  N),
                 ResidualBlock(N,  N),
                )

        self.enc_base_rep = Seq(
                 ResidualBlock(N,N),
                 ResidualBlock(N,N)
                
                )
            
        self.enc_p0_rep = Seq(
                ResidualBlock(N,N),
                ResidualBlock(N,N)
                
                )
            

        
        if self.portion == "small":
            self.enc = Seq(
                ResidualBlockSmall(4 * N,2*N),
                ResidualBlockSmall(2 * N,  N)

            )
        else: 
            self.enc = Seq(
                ResidualBlock(4 * N,2*N),
                ResidualBlock(2 * N,  N)

            )           


    def extract_map(self,x,net):

        out_base = net.forward_single_quality(x, 
                                                  quality = 0, 
                                                  training = False, 
                                                  force_enhanced = False,
                                                  mask_pol = "point-based-std")
        
        out_p0 = net.forward_single_quality(x, 
                                                  quality = 0.001, 
                                                  training = False, 
                                                  force_enhanced = True,
                                                  mask_pol = "point-based-std")
        

        y_hat_b = out_base["y_hat"]
        mu_b = out_base["mu"]
        std_b = out_base["std"]

        y_hat_p0 = out_p0["y_hat"]
        mu_p0 = out_p0["mu"]
        std_p0 = out_p0["std"]


        params_b = torch.cat([mu_b,std_b],dim = 1)
        params_p0 = torch.cat([mu_p0,std_p0],dim = 1)


        mask = self.forward(y_hat_b,y_hat_p0,params_b,params_p0)
        return mask



    def forward(self, y_base, y_p0, entropy_params_base, entropy_params_p0):

        f_latent_b = self.enc_base_rep(y_base)
        f_latent_p0 = self.enc_p0_rep(y_p0)
        f_ent_base = self.enc_base_entropy_params(entropy_params_base)
        f_ent_prog = self.enc_p0_entropy_params(entropy_params_p0) 
        ret = self.enc(torch.cat([f_latent_b,f_latent_p0, f_ent_base, f_ent_prog], dim=1))

        if self.normalize:
            ret = torch.sigmoid(ret) 

        #ret = (ret - ret.min()) / (ret.max() - ret.min())
        return ret
        

    def print_information(self):

        print("self.enc_base_rep",sum(p.numel() for p in self.enc_base_rep.parameters()))
        print("self.enc_enh_rep",sum(p.numel() for p in self.enc_p0_rep.parameters()))  
        print("self.enc_base_rep",sum(p.numel() for p in self.enc_base_entropy_params.parameters()))  
        print("self.enc_enh_entropy_params",sum(p.numel() for p in self.enc_p0_entropy_params.parameters()))    
        print("self.enc",sum(p.numel() for p in self.enc.parameters())) 
        print("**************************************************************************")
        model_tr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_fr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad== False)
        print(" trainable parameters: ",model_tr_parameters)
        print(" freeze parameterss: ", model_fr_parameters)
        return model_tr_parameters
    


class PostRateProcessedNetwork(nn.Module):

    def __init__(self, 
                base_net,
                check_levels = [0.01,0.25,1.75],
                mu_std = False,
                dimension = "big",
                escalation = False
                    ):
        super().__init__()


        self.base_net = base_net
        self.mu_std = mu_std


        self.check_levels = check_levels
        self.check_multiple = len(self.check_levels)

        assert isinstance(self.base_net, ChannelProgresssiveWACNN)

        self.escalation = escalation

        self.post_latent = nn.ModuleList(
                                nn.ModuleList( 
                                                LatentRateReduction(dim_chunk = self.base_net.dim_chunk,
                                                 mu_std = self.mu_std, dimension=dimension) 
                                                for _ in range(10))
                        for _ in range(self.check_multiple)
                    )

        print("questi sono i livelli: ",self.check_levels," ",len(self.post_latent))
        


    def print_information(self, type = "all"):
        if type == "all":
            print("base net",sum(p.numel() for p in self.base_net.parameters()))
            print("base trainable net",sum(p.numel() for p in self.base_net.parameters() if p.requires_grad is True) )

            print("post net",sum(p.numel() for p in self.post_latent.parameters())) 
            print("post trainable net",sum(p.numel() for p in self.post_latent.parameters() if p.requires_grad is True)) 
            return  sum(p.numel() for p in self.post_latent.parameters() if p.requires_grad is True)\
                    + sum(p.numel() for p in self.base_net.parameters() if p.requires_grad is True)
        elif type == "decoder":
            post_net_p = sum(p.numel() for p in self.post_latent.parameters())
            print("post net",post_net_p) 
            g_s_p = sum(p.numel() for p in self.base_net.g_s.parameters())
            cc_mean = sum(p.numel() for p in self.base_net.cc_mean_transforms.parameters()) 
            cc_scale = sum(p.numel() for p in self.base_net.cc_scale_transforms.parameters())
            h_mean = sum(p.numel() for p in self.base_net.h_mean_s.parameters())
            h_scale = sum(p.numel() for p in self.base_net.h_scale_s.parameters())
            cc_mean_prog = sum(p.numel() for p in self.base_net.cc_mean_transforms_prog.parameters())
            cc_scale_prog = sum(p.numel() for p in self.base_net.cc_scale_transforms_prog.parameters())
            lrp = sum(p.numel() for p in self.base_net.lrp_transforms.parameters())
            print("g_s: ",g_s_p) 
            print("cc_mean_transforms",cc_mean)
            print("cc_scale_transforms",cc_scale)
            print(" h_means_a: ",h_mean)
            print(" h_scale_a: ",h_scale)
            print("cc_mean_transforms_prog",cc_mean_prog)
            print("cc_scale_transforms_prog",cc_scale_prog)  
            print("lrp_transform",lrp)
            total = lrp + cc_scale_prog + cc_mean_prog + h_scale + h_mean + cc_scale + cc_mean
            return total
        elif type == "encoder":
            post_net_p = sum(p.numel() for p in self.post_latent.parameters())
             
            g_s_p = sum(p.numel() for p in self.base_net.g_a.parameters())
            cc_mean = sum(p.numel() for p in self.base_net.cc_mean_transforms.parameters()) 
            cc_scale = sum(p.numel() for p in self.base_net.cc_scale_transforms.parameters())
            h_mean = sum(p.numel() for p in self.base_net.h_a.parameters())
            
            cc_mean_prog = sum(p.numel() for p in self.base_net.cc_mean_transforms_prog.parameters())
            cc_scale_prog = sum(p.numel() for p in self.base_net.cc_scale_transforms_prog.parameters())
            lrp = sum(p.numel() for p in self.base_net.lrp_transforms.parameters()) - sum(p.numel() for p in self.base_net.lrp_transforms[-1].parameters())
            
            print("post net",post_net_p)
            print("g_a: ",g_s_p) 
            print("cc_mean_transforms",cc_mean)
            print("cc_scale_transforms",cc_scale)
            print(" h_means_a: ",h_mean)
            print("cc_mean_transforms_prog",cc_mean_prog)
            print("cc_scale_transforms_prog",cc_scale_prog)  
            print("lrp_transform",lrp)


            total = g_s_p + post_net_p + lrp + cc_scale_prog + cc_mean_prog +  h_mean + cc_scale + cc_mean 
            print("TOTAL: ",total)
            return  total




    def freeze(self):
        for p in self.base_net.parameters():
            p.requires_grad = False

        for p in self.post_latent.parameters():
            p.requires_grad = True 
    
    def unfreeze_g_s(self,also_base_dec = False):

        for p in self.parameters():
            p.requires_grad = False
        
        for p in self.base_net.parameters(): #fff
            p.requires_grad = False

        if self.base_net.multiple_decoder is False:
            for p in self.base_net.g_s.parameters():
                p.requires_grad = True
        else:          
            
            for p in self.base_net.g_s[1].parameters():
                p.requires_grad = True 
            
            if also_base_dec:
                for p in self.base_net.g_s[0].parameters():
                    p.requires_grad = True 


    def load_state_dict(self, state_dict_base,state_dict_post = None, strict = False):
        self.base_net.load_state_dict(state_dict_base,strict = strict)

        if state_dict_post is not None:
            print("load also post_latent")
            self.post_latent.load_state_dict(state_dict_post,strict = strict)
        else:
            print("initially they are None")


    def extract_chekpoint_representation_from_images(self,x, quality,  rc = True): #fff

        if self.escalation is False: #dddd
            out_latent = self.compress( x, 
                                   quality =quality,
                                    mask_pol ="point-based-std",
                                    real_compress=rc)#["y_hat"] #ddddddddddddd

            return out_latent["y_hat"]
        else:
            out_latent = self.compress( x, 
                                   quality =self.check_levels[0],
                                    mask_pol ="point-based-std",
                                    real_compress=rc)["y_hat"] #ddd
            
            if quality == self.check_levels[0]:
                return out_latent 
            
            out_latent_1 = self.compress( x, 
                                   quality =self.check_levels[1],
                                    mask_pol ="point-based-std",
                                    checkpoint_rep= out_latent,
                                    real_compress=rc)["y_hat"] #ddd
            
            if quality == self.check_levels[1]:
                return out_latent_1 
            


            out_latent_2 = self.compress( x, 
                                   quality =self.check_levels[2],
                                    mask_pol ="point-based-std",
                                    checkpoint_rep= out_latent_1,
                                    real_compress=rc)["y_hat"] #ddd
            
            return out_latent_2


    def apply_latent_enhancement(self,
                                current_index,
                                quality,
                                quality_bar,
                                y_b_hat, 
                                mu_scale_base, 
                                mu_scale_enh,
                                mu, 
                                scale,
                                training = False,
                                mask_pol = "point-based-std",
                                attention_mask = None):


        if attention_mask is None:
            bar_mask = self.base_net.masking(scale,
                                      slice_index = current_index, 
                                      pr = quality_bar,
                                      mask_pol = mask_pol)
            star_mask = self.base_net.masking(scale,
                                      slice_index = current_index, 
                                      pr = quality,
                                      mask_pol = mask_pol) 

            attention_mask = star_mask - bar_mask 
            attention_mask = self.base_net.masking.apply_noise(attention_mask, training)   

        if self.mu_std:
            attention_mask = torch.cat([attention_mask,attention_mask],dim = 1)  
        # in any case I do not improve anithing here!
        if quality <= self.check_levels[0]: # l'unico che c'è  CAMBIAREEEEEE
            return mu, scale         

        if self.check_multiple == 1:
            enhanced_params =  self.post_latent[0][current_index](y_b_hat, mu_scale_base, mu_scale_enh, attention_mask)
        elif self.check_multiple == 2:
            index = 0 if self.check_levels[0] < quality <= self.check_levels[1] else 1 
            enhanced_params =  self.post_latent[index][current_index](y_b_hat, mu_scale_base, mu_scale_enh, attention_mask)
        else: 
            index = -1 
            if self.check_levels[0] < quality <= self.check_levels[1]: #ffff
                index = 0 
                #index = -1
            elif  self.check_levels[1] < quality <= self.check_levels[2]:
                index = 1
            else:
                index = 2 
            enhanced_params =  self.post_latent[index][current_index](y_b_hat, mu_scale_base, mu_scale_enh, attention_mask)   
        if self.mu_std:
                mu,scale = enhanced_params.chunk(2,1)
                return mu, scale
        else:
            scale = enhanced_params
            return mu, scale







        

    def forward(self, x, mask_pol = "point-based-std", quality = None, training  = True, checkpoint_ref = None ,write_stats = False):
        
        out_latent = self.forward_latent(x,mask_pol = mask_pol, quality=quality,training=training, checkpoint_ref = checkpoint_ref, write_stats=write_stats)
        y_hat = out_latent["y_hat"] #sss
        index = 0 if quality == 0 else 1
        x_hat = self.base_net.g_s[index](y_hat).clamp_(0, 1) if self.base_net.multiple_decoder  \
                else self.base_net.g_s(y_hat).clamp_(0, 1)
        out_latent["x_hat"] = x_hat         
        return out_latent


    def find_check_quality(self,quality):
        if quality <= self.check_levels[0]:
            quality_ref = 0 
            quality_post = 0

        elif (len(self.check_levels) == 2 or len(self.check_levels) == 3)  and self.check_levels[0] < quality <= self.check_levels[1]:
                quality_ref = self.check_levels[0]
                quality_post = self.check_levels[1]
        elif len(self.check_levels) == 2 and quality > self.check_levels[1]:
            quality_ref = self.check_levels[1]
            quality_post = 10
        
        elif len(self.check_levels)==3 and  self.check_levels[1] < quality <= self.check_levels[2]:
            quality_ref = self.check_levels[1] 
            quality_post = self.check_levels[-1]
        else:
            quality_ref = self.check_levels[-1]
            quality_post  = 10
        return quality_ref, quality_post

    def forward_latent(self, x, mask_pol, quality, training, checkpoint_ref = None, write_stats = False):
        quality = self.starting_quality if quality is None else quality 
        mask_pol = self.mask_policy if mask_pol is None else mask_pol

        if self.base_net.multiple_encoder is False:
            y = self.base_net.g_a(x)
            y_base = y 
            y_enh = y
        else:
            y_base = self.base_net.g_a[0](x)
            y_enh = self.base_net.g_a[1](x)
            y = torch.cat([y_base,y_enh],dim = 1).to(x.device) #dddd
        y_shape = y.shape[2:]
        latent_means, latent_scales, z_likelihoods = self.base_net.compute_hyperprior(y, quality)

        y_slices = y.chunk(self.base_net.num_slices, 1) # total amount of slicesy,

        y_hat_slices = []
        y_likelihood = []

        mu_base, mu_prog = [],[]
        std_base, std_prog = [],[]

        for slice_index in range(self.base_net.num_slice_cumulative_list[0]):
            y_slice = y_slices[slice_index]
            idx = slice_index%self.base_net.num_slice_cumulative_list[0]
            indice = min(self.base_net.max_support_slices,idx)
            support_slices = (y_hat_slices if self.base_net.max_support_slices < 0 else y_hat_slices[:indice]) 
            
            mean_support = torch.cat([latent_means[:,:self.base_net.division_dimension[0]]] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales[:,:self.base_net.division_dimension[0]]] + support_slices, dim=1) 

            
            mu = self.base_net.cc_mean_transforms[idx](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]  
            scale = self.base_net.cc_scale_transforms[idx](scale_support)#self.extract_scale(idx,slice_index,scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            mu_base.append(mu)
            std_base.append(scale) 

            mu_prog.append(mu) #le sommo
            std_prog.append(scale) #le sommo 

            _, y_slice_likelihood = self.base_net.gaussian_conditional(y_slice, scale, mu, training = training)
            y_hat_slice = ste_round(y_slice - mu) + mu

            lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
            lrp = self.base_net.lrp_transforms[idx](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp               

            y_hat_slices.append(y_hat_slice)
            y_likelihood.append(y_slice_likelihood)

        if quality == 0: #and  slice_index == self.num_slice_cumulative_list[0] - 1:
            y_hat = torch.cat(y_hat_slices,dim = 1)
            #x_hat = self.g_s[0](y_hat).clamp_(0, 1) if self.multiple_decoder else self.g_s(y_hat).clamp_(0, 1)
            y_likelihoods = torch.cat(y_likelihood, dim=1)
            return {
                "likelihoods": {"y": y_likelihoods,"z": z_likelihoods},
            "y_hat":y_hat,"y_base":y_hat,"y_complete":y_hat,
            "mu_base":mu_base,"mu_prog":mu_prog,"std_base":std_base,"std_prog":std_prog

            }             
        y_hat_b = torch.cat(y_hat_slices,dim = 1)
        y_hat_slices_quality = []
        

        y_likelihood_quality = []
        y_likelihood_quality = y_likelihood + []#ffff

        y_checkpoint_hat = checkpoint_ref.chunk(10,1) if checkpoint_ref is not None else y_hat_slices


        mu_total = []
        std_total = []


        for slice_index in range(self.base_net.ns0,self.base_net.ns1):

            y_slice = y_slices[slice_index]
            current_index = slice_index%self.base_net.ns0


            if self.base_net.delta_encode:
                y_slice = y_slice - y_slices[current_index] 





            support_vector = mu_total if self.base_net.all_scalable else y_hat_slices_quality
            support_vector_std = std_total if self.base_net.all_scalable else y_hat_slices_quality
            support_slices_mean = self.base_net.determine_support(y_hat_slices,
                                                         current_index,
                                                        support_vector                                                      
                                                         )
            support_slices_std = self.base_net.determine_support(y_hat_slices,
                                                         current_index,
                                                        support_vector_std                                                      
                                                         )


            
            mean_support = torch.cat([latent_means[:,self.base_net.dimensions_M[0]:]] + support_slices_mean, dim=1)
            scale_support = torch.cat([latent_scales[:,self.base_net.dimensions_M[0]:]] + support_slices_std, dim=1) 

            mu = self.base_net.cc_mean_transforms_prog[current_index](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
            mut = mu + y_hat_slices[current_index] if self.base_net.total_mu_rep else mu
            mu = mu[:, :, :y_shape[0], :y_shape[1]]  

            scale = self.base_net.cc_scale_transforms_prog[current_index](scale_support)#self.extract_scale(idx,slice_index,scale_support)
            

            mu_prog[current_index] = mu_prog[current_index] + mu
            std_prog[current_index] = std_prog[current_index] +  scale 


            
            if self.base_net.support_std:
                std_total.append(scale)
            else:
                std_total.append(mut)

            mu_total.append(mut)

            scale = scale[:, :, :y_shape[0], :y_shape[1]] #fff


            # qua avviene la magia! 
            mu_scale_base = torch.cat([mu_base[current_index],std_base[current_index]],dim = 1) 
            mu_scale_enh =  torch.cat([mu,scale],dim = 1) if self.mu_std else scale



            #block_mask = self.base_net.masking(scale,
            #                          scale_base = None ,
            #                          slice_index = current_index, 
            #                          pr = quality,
            #                          mask_pol = mask_pol) 
            #block_mask = self.base_net.masking.apply_noise(block_mask, training)
            
            
            y_b_hat = y_checkpoint_hat[current_index]
            y_b_hat.requires_grad = True

            quality_bar, _  = self.find_check_quality(quality)



            mu, scale = self.apply_latent_enhancement(current_index,
                                                        quality,
                                                        quality_bar,
                                                        y_b_hat, 
                                                        mu_scale_base, 
                                                        mu_scale_enh,
                                                        mu, 
                                                        scale,
                                                        training =False)
            


            block_mask = self.base_net.masking(scale,
                                      scale_base = None ,
                                      slice_index = current_index, 
                                      pr = quality,
                                      mask_pol = mask_pol) 
            block_mask = self.base_net.masking.apply_noise(block_mask, training)



            y_slice_m = (y_slice  - mu)*block_mask
            _, y_slice_likelihood = self.base_net.gaussian_conditional(y_slice_m, scale*block_mask, training = training)
            y_hat_slice = ste_round(y_slice - mu)*block_mask + mu


            y_likelihood_quality.append(y_slice_likelihood)


            lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
            lrp = self.base_net.lrp_transforms_prog[current_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp   

            y_hat_slice = self.base_net.merge(y_hat_slice,y_hat_slices[current_index],current_index)   #ddd
            y_hat_slices_quality.append(y_hat_slice) 
 

        y_likelihoods = torch.cat(y_likelihood_quality,dim = 1) #ddddd
        y_hat_p = torch.cat(y_hat_slices_quality,dim = 1) 

        mu_base = torch.cat(mu_base,dim = 1)
        mu_prog = torch.cat(mu_prog,dim = 1)
        std_base = torch.cat(std_base,dim = 1)   
        std_prog = torch.cat(std_prog, dim = 1)#kkkk

        return {
            "likelihoods": {"y": y_likelihoods,"z": z_likelihoods},
            "y_hat":y_hat_p,"y_base":y_hat_b,
            "mu_base":mu_base,"mu_prog":mu_prog,"std_base":std_base,"std_prog":std_prog
        }     


    def compress(self, x, 
                 quality = 0.0, 
                 mask_pol = "point-based-std", 
                 checkpoint_rep = None, 
                 real_compress = True, 
                 used_qual = None):

        used_qual = self.check_levels if used_qual is None else used_qual


        if self.base_net.multiple_encoder is False:
            y = self.base_net.g_a(x)
        else:
            y_base = self.base_net.g_a[0](x)
            y_enh = self.base_net.g_a[1](x)
            y = torch.cat([y_base,y_enh],dim = 1).to(x.device)
        y_shape = y.shape[2:]

        z = self.base_net.h_a(y)
        z_strings =  self.base_net.entropy_bottleneck.compress(z)
        
        z_hat = self.base_net.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        latent_scales = self.base_net.h_scale_s(z_hat) if self.base_net.multiple_hyperprior is False \
                        else self.base_net.h_scale_s[0](z_hat)
        latent_means = self.base_net.h_mean_s(z_hat) if self.base_net.multiple_hyperprior is False \
                        else self.base_net.h_mean_s[0](z_hat)

        if self.base_net.multiple_hyperprior and quality > 0:
            latent_scales_enh = self.base_net.h_scale_s[1](z_hat) 
            latent_means_enh = self.base_net.h_mean_s[1](z_hat)
            latent_means = torch.cat([latent_means,latent_means_enh],dim = 1)
            latent_scales = torch.cat([latent_scales,latent_scales_enh],dim = 1) 

        y_hat_slices = []

        y_slices = y.chunk(self.base_net.num_slices, 1) # total amount of slices
        y_strings = []
        masks = []
        mu_base = []
        std_base = []

        for slice_index in range(self.base_net.ns0):
            y_slice = y_slices[slice_index]
            indice = min(self.base_net.max_support_slices,slice_index%self.base_net.ns0)


            support_slices = (y_hat_slices if self.base_net.max_support_slices < 0 \
                                                        else y_hat_slices[:indice])               
            
            idx = slice_index%self.base_net.ns0


            mean_support = torch.cat([latent_means[:,:self.base_net.division_dimension[0]]] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales[:,:self.base_net.division_dimension[0]]] + support_slices, dim=1) 

            
            mu = self.base_net.cc_mean_transforms[idx](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]  
            scale = self.base_net.cc_scale_transforms[idx](scale_support)#self.extract_scale(idx,slice_index,scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]


            #print("mssimo e minimo base----> ",torch.min(scale)," ",torch.max(scale))

            mu_base.append(mu) 
            std_base.append(scale)

            index = self.base_net.gaussian_conditional.build_indexes(scale)
            if real_compress:
                y_q_string  = self.base_net.gaussian_conditional.compress(y_slice, index,mu)
                y_hat_slice = self.base_net.gaussian_conditional.decompress(y_q_string, index)
                y_hat_slice = y_hat_slice + mu
            else:
                y_q_string  = self.base_net.gaussian_conditional.quantize(y_slice, "symbols", mu)#ddd
                y_hat_slice = y_q_string + mu

            y_strings.append(y_q_string)

            lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
            lrp = self.base_net.lrp_transforms[idx](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        if quality <= 0:
            return {"strings": [y_strings, z_strings],
                    "shape":z.size()[-2:], 
                    "masks":masks,
                    "y":y,
                    "y_hat":torch.cat(y_hat_slices,dim = 1),
                    "latent_means":latent_means,
                    "latent_scales":latent_scales,
                    "mean_base":torch.cat(mu_base,dim = 1),
                    "std_base":torch.cat(std_base,dim = 1)}
        
        y_hat_slices_quality = []



        y_b_hats = checkpoint_rep.chunk(10,1) if checkpoint_rep is not None else y_hat_slices 

        mu_total, std_total = [],[]
        for slice_index in range(self.base_net.ns0,self.base_net.ns1): #ffff

            y_slice = y_slices[slice_index]
            current_index = slice_index%self.base_net.ns0

            if self.base_net.delta_encode:
                y_slice = y_slice - y_slices[current_index] 


            support_vector = mu_total if self.base_net.all_scalable else y_hat_slices_quality
            support_vector_std = std_total if self.base_net.all_scalable else y_hat_slices_quality
            support_slices_mean = self.base_net.determine_support(y_hat_slices,
                                                         current_index,
                                                        support_vector                                                      
                                                         )
            support_slices_std = self.base_net.determine_support(y_hat_slices,
                                                         current_index,
                                                        support_vector_std                                                      
                                                         )


            
            mean_support = torch.cat([latent_means[:,self.base_net.dimensions_M[0]:]] + support_slices_mean, dim=1)
            scale_support = torch.cat([latent_scales[:,self.base_net.dimensions_M[0]:]] + support_slices_std, dim=1) 

            mu = self.base_net.cc_mean_transforms_prog[current_index](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
            mut = mu + y_hat_slices[current_index] if self.base_net.total_mu_rep else mu
            mu = mu[:, :, :y_shape[0], :y_shape[1]]  

            scale = self.base_net.cc_scale_transforms_prog[current_index](scale_support)#self.extract_scale(idx,slice_index,scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]] #fff
            if  self.base_net.support_std:
                std_total.append(scale)
            else:
                std_total.append(mut)

            mu_total.append(mut)


            
            y_b_hat = y_b_hats[current_index]

            mu_scale_base = torch.cat([mu_base[current_index],std_base[current_index]],dim = 1) 
            mu_scale_enh =  torch.cat([mu,scale],dim = 1) if self.mu_std else scale
            
            
            
            quality_bar,quality_post = self.find_check_quality(quality)


            #block_mask = self.base_net.masking(scale,
            #                        slice_index = current_index,
            #                        pr = quality, 
            #                        mask_pol = mask_pol) 
            #print("mssimo e minimo top----> ",torch.min(scale)," ",torch.max(scale))#ddd

            mu, scale = self.apply_latent_enhancement(current_index, 
                                                      quality, 
                                                      quality_bar,
                                                   
                                                      y_b_hat, 
                                                      mu_scale_base,
                                                        mu_scale_enh,
                                                        mu,
                                                        scale,
                                                        training = False)

            block_mask = self.base_net.masking(scale,
                                    slice_index = current_index,
                                    pr = quality, 
                                    mask_pol = mask_pol) 

            """
            if write_stats and current_index <= 1:
                print(quality_bar)
                print("compress: ",current_index,"   ",quality)
                print(quality_bar)
                print("scale: ",torch.unique(scale,return_counts = True))
                print("mean: ",torch.unique(mu,return_counts = True))
                print("------------------------------------------------------")
            """
            
            masks.append(block_mask)
            block_mask = self.base_net.masking.apply_noise(block_mask, False)
            index = self.base_net.gaussian_conditional.build_indexes(scale*block_mask).int() 
            if real_compress:
                #y_hat_zero = self.base_net.gaussian_conditional.quantize((y_slice - mu)*block_mask,"symbols")
                y_q_string  = self.base_net.gaussian_conditional.compress((y_slice - mu)*block_mask, index)
                y_strings.append(y_q_string)
    
                y_hat_slice_nomu = self.base_net.gaussian_conditional.decompress(y_q_string, index)
                y_hat_slice = y_hat_slice_nomu + mu

                #y_q_string  = self.base_net.gaussian_conditional.quantize((y_slice - mu)*block_mask, "dequantize")
                #y_hat_slice = y_q_string + mu

            else:
                y_q_string  = self.base_net.gaussian_conditional.quantize((y_slice - mu)*block_mask, "dequantize") #ffff
                y_strings.append(y_q_string)
                y_hat_slice = y_q_string + mu


            lrp_support = torch.cat([mean_support,y_hat_slice], dim=1)
            lrp = self.base_net.lrp_transforms_prog[current_index](lrp_support) #ddd
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            if self.base_net.residual_before_lrp is False:
                y_hat_slice = self.base_net.merge(y_hat_slice,y_hat_slices[current_index],current_index)

            y_hat_slices_quality.append(y_hat_slice)

        return {"strings": [y_strings, z_strings],"shape":z.size()[-2:],"masks":masks,"y_hat":torch.cat(y_hat_slices_quality,dim = 1)}


    def decompress_check_rep(self,strings,shape, quality):#dddd
        return self.decompress(strings,shape, quality,mask_pol="point-based-std",timing = False)["y_hat"]



    def decompress(self, 
                   strings,
                    shape, 
                    quality,
                    mask_pol = None,
                    checkpoint_rep = None, 
                    timing = False,
                    used_qual = None):
        

        used_qual = self.check_levels if used_qual is None else used_qual
        mask_pol = self.base_net.mask_policy if mask_pol is None else mask_pol

        if timing:
            start_t = time.time()


        z_hat = self.base_net.entropy_bottleneck.decompress(strings[1], shape)
        latent_scales = self.base_net.h_scale_s(z_hat) if self.base_net.multiple_hyperprior is False    \
                        else self.base_net.h_scale_s[0](z_hat)
        latent_means = self.base_net.h_mean_s(z_hat) if self.base_net.multiple_hyperprior is False \
                        else self.base_net.h_mean_s[0](z_hat)

    
        if self.base_net.multiple_hyperprior and quality > 0:
            latent_scales_enh = self.base_net.h_scale_s[1](z_hat) 
            latent_means_enh = self.base_net.h_mean_s[1](z_hat)
            latent_means = torch.cat([latent_means,latent_means_enh],dim = 1)
            latent_scales = torch.cat([latent_scales,latent_scales_enh],dim = 1) 

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        y_string = strings[0]
        y_hat_slices = []


        mu_base = []
        std_base = []


        for slice_index in range(self.base_net.num_slice_cumulative_list[0]): #ddd
            pr_strings = y_string[slice_index]
            idx = slice_index%self.base_net.num_slice_cumulative_list[0]
            indice = min(self.base_net.max_support_slices,idx)
            support_slices = (y_hat_slices if self.base_net.max_support_slices < 0 else y_hat_slices[:indice]) 
            
            mean_support = torch.cat([latent_means[:,:self.base_net.division_dimension[0]]] + support_slices, dim=1)
            scale_support = torch.cat([latent_scales[:,:self.base_net.division_dimension[0]]] + support_slices, dim=1) 
      
            mu = self.base_net.cc_mean_transforms[idx](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
            mu = mu[:, :, :y_shape[0], :y_shape[1]]  
            scale = self.base_net.cc_scale_transforms[idx](scale_support)#self.extract_scale(idx,slice_index,scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            mu_base.append(mu)
            std_base.append(scale)

            index = self.base_net.gaussian_conditional.build_indexes(scale)


            rv = self.base_net.gaussian_conditional.decompress(pr_strings, index )
            rv = rv.reshape(mu.shape)
            y_hat_slice = self.base_net.gaussian_conditional.dequantize(rv, mu)

            

            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.base_net.lrp_transforms[idx](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            y_hat_slices.append(y_hat_slice)

        if quality == 0:
            y_hat_b = torch.cat(y_hat_slices, dim=1)

            
            if timing:
                end_t = time.time()
                time_ = end_t - start_t
            else:
                time_ = 0
            x_hat = self.base_net.g_s[0](y_hat_b).clamp_(0, 1) if self.base_net.multiple_decoder else \
                    self.base_net.g_s(y_hat_b).clamp_(0, 1)
            return {"x_hat": x_hat, "y_hat": y_hat_slices,"time":time_}



        if timing:
            start_t = time.time()

        y_hat_slices_quality = []


        y_b_hats = checkpoint_rep.chunk(10,1) if checkpoint_rep is not None else y_hat_slices  
        mu_total,std_total = [],[]

        
        for slice_index in range(self.base_net.ns0,self.base_net.ns1):
            pr_strings = y_string[slice_index]
            current_index = slice_index%self.base_net.ns0




            support_vector = mu_total if self.base_net.all_scalable else y_hat_slices_quality
            support_vector_std = std_total if self.base_net.all_scalable else y_hat_slices_quality
            support_slices_mean = self.base_net.determine_support(y_hat_slices,
                                                         current_index,
                                                        support_vector                                                      
                                                         )
            support_slices_std = self.base_net.determine_support(y_hat_slices,
                                                         current_index,
                                                        support_vector_std                                                      
                                                         )


            
            mean_support = torch.cat([latent_means[:,self.base_net.dimensions_M[0]:]] + support_slices_mean, dim=1)
            scale_support = torch.cat([latent_scales[:,self.base_net.dimensions_M[0]:]] + support_slices_std, dim=1) 

            mu = self.base_net.cc_mean_transforms_prog[current_index](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
            mut = mu + y_hat_slices[current_index] if self.base_net.total_mu_rep else mu
            mu = mu[:, :, :y_shape[0], :y_shape[1]]  

            scale = self.base_net.cc_scale_transforms_prog[current_index](scale_support)#self.extract_scale(idx,slice_index,scale_support)
            


            
            if self.base_net.support_std:
                std_total.append(scale)
            else:
                std_total.append(mut)

            mu_total.append(mut)

            scale = scale[:, :, :y_shape[0], :y_shape[1]] #fff

            y_b_hat =  y_b_hats[current_index]

            #y_b_hat = y_b_hats[current_index]
            mu_scale_base = torch.cat([mu_base[current_index],std_base[current_index]],dim = 1) 
            mu_scale_enh =  torch.cat([mu,scale],dim = 1) if self.mu_std else scale


            
            

            #y_b_hat = y_b_hats[current_index]
            mu_scale_base = torch.cat([mu_base[current_index],std_base[current_index]],dim = 1) 
            mu_scale_enh =  torch.cat([mu,scale],dim = 1) if self.mu_std else scale


            
            quality_bar,quality_post = self.find_check_quality(quality)


            #block_mask = self.base_net.masking(scale,
            #                        slice_index = current_index,
            #                        pr = quality, 
            #                        mask_pol = mask_pol) 



            mu, scale = self.apply_latent_enhancement(current_index, 
                                                      quality, 
                                                      quality_bar,
                                                     
                                                      y_b_hat, 
                                                      mu_scale_base,
                                                        mu_scale_enh,
                                                        mu,
                                                        scale,
                                                        False)



            block_mask = self.base_net.masking(scale,
                                    slice_index = current_index,
                                    pr = quality, 
                                    mask_pol = mask_pol) 



            block_mask_bis = self.base_net.masking(scale,
                                    slice_index = current_index,
                                    pr = quality, 
                                    mask_pol = mask_pol) 
 

   

            diff_mask = block_mask - block_mask_bis
            mubis = mu*diff_mask
            stdmask = scale*diff_mask 


            index = self.base_net.gaussian_conditional.build_indexes(scale*block_mask)
            rv = self.base_net.gaussian_conditional.decompress(pr_strings, index)
            rv = rv.reshape(mu.shape)

            y_hat_slice = rv + mu #self.base_net.gaussian_conditional.dequantize(rv, mu)




            lrp_support = torch.cat([mean_support, y_hat_slice], dim=1)
            lrp = self.base_net.lrp_transforms_prog[current_index](lrp_support)
            lrp = 0.5 * torch.tanh(lrp)
            y_hat_slice += lrp

            
            if self.base_net.residual_before_lrp is False:
                y_hat_slice = self.base_net.merge(y_hat_slice,y_hat_slices[current_index],current_index)

            y_hat_slices_quality.append(y_hat_slice)


        y_hat_en = torch.cat(y_hat_slices_quality,dim = 1)

        if timing:
            end_t = time.time()
            time_ = end_t - start_t
        else:
            time_ = 0
        #y_hat_nolrp = torch.cat( y_hat_slices_nolrp,dim = 1)
        if self.base_net.multiple_decoder:
            x_hat = self.base_net.g_s[1](y_hat_en).clamp_(0, 1)
        else:
            x_hat = self.base_net.g_s(y_hat_en).clamp_(0, 1) 
        return {"x_hat": x_hat,"y_hat":y_hat_en,"time":time_}   



    def find_init_masking(self,current_index,latent_means,latent_scales,y_hat_slices, y_hat_init_slices, y_shape):
        support_slices = self.base_net.determine_support(y_hat_slices,current_index,y_hat_init_slices)
        scale_support = torch.cat([latent_scales[:,self.base_net.division_dimension[0]:]] + support_slices, dim=1) 
        scale = self.base_net.cc_scale_transforms_prog[current_index](scale_support)#self.extract_scale(idx,slice_index,scale_support)
        scale = scale[:, :, :y_shape[0], :y_shape[1]]

        mean_support = torch.cat([latent_means[:,self.base_net.division_dimension[0]:]] + support_slices, dim=1)
        mu = self.base_net.cc_mean_transforms_prog[current_index](mean_support)  #self.extract_mu(idx,slice_index,mean_support)
        mu = mu[:, :, :y_shape[0], :y_shape[1]]  
        return mu,scale



