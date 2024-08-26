
import torch.nn as nn
from compress.layers import GDN, Win_noShift_Attention
import torch
from torch.nn import Sequential as Seq
from torch.autograd import Variable

def conv(in_channels, out_channels, kernel_size=5, stride=2):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def deconv(in_channels, out_channels, kernel_size=5, stride=2):     # SN -1 + k - 2p
    return nn.ConvTranspose2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        output_padding=stride - 1,
        padding=kernel_size // 2,
    )


         





def conv3x3(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1)


def conv1x1(in_ch: int, out_ch: int, stride: int = 1) -> nn.Module:
    return nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0)#eeee


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)

  
        self.nonlin = nn.LeakyReLU(inplace=True)

        
        self.conv2 = conv3x3(out_ch, out_ch)

        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.nonlin(out)
        out = self.conv2(out)
        out = self.nonlin(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out
    



class ResidualBlockSmall(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)

  
        self.nonlin = nn.LeakyReLU(inplace=True)

        
        #self.conv2 = conv3x3(out_ch, out_ch)

        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.nonlin(out)
        #out = self.conv2(out)
        #out = self.nonlin(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out



class ResidualBlockGDN(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, inverse = False):
        super().__init__()
        self.conv1 = conv3x3(in_ch, out_ch)
        self.nonlin = GDN(out_ch,inverse=inverse)
        self.conv2 = conv3x3(out_ch, out_ch)

        if in_ch != out_ch:
            self.skip = conv1x1(in_ch, out_ch)
        else:
            self.skip = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.nonlin(out)

        if self.skip is not None:
            identity = self.skip(x)

        out = out + identity
        return out



class LatentPostNet(nn.Module):
    def __init__(self, N):
        super().__init__()

        self.enc_latent = Seq(
            ResidualBlock(N, N),
            ResidualBlock(N, N),
        )

        self.enc_entropy_params = Seq(
            ResidualBlock(2 * N, 2 * N),
            ResidualBlock(2 * N, N),
        )

        self.enc = Seq(
            ResidualBlock(2 * N, 2 * N),
            ResidualBlock(2 * N, N),
        )

    def forward(self, x, entropy_params):
        identity = x

        f_latent = self.enc_latent(x)
        f_ent = self.enc_entropy_params(entropy_params)

        ret = self.enc(torch.cat([f_latent, f_ent], dim=1))

        ret += identity


        return ret
    



class PostNet(nn.Module):

    def __init__(self, N = 128,M = 192,model_type = 0, **kwargs):
        super().__init__( **kwargs)


        assert model_type == 0 or model_type == 1
        self.model_type = model_type

        if self.model_type == 0:
            self.res1 = ResidualBlockGDN(3,N)
            self.res2 = ResidualBlockGDN(N,N)
            self.attn1 = Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4)
            self.res3 = ResidualBlockGDN(N,N)
            self.res4 = ResidualBlockGDN(N,M)

            self.res5 = ResidualBlockGDN(M,N,inverse=True) 
            self.attn2 = Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4)
            self.res6 = ResidualBlockGDN(N,N,inverse=True) 
            self.res7 = ResidualBlockGDN(N,N,inverse=True) 
            self.res8 = ResidualBlock(N,3) 
        else:
            self.post_encoder = nn.Sequential(
                conv(3, N, kernel_size=5, stride=2), # halve 128
                GDN(N),
                conv(N, N, kernel_size=5, stride=2), # halve 64
                GDN(N),
                Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4), # 
                conv(N, N, kernel_size=5, stride=2), #32 
                GDN(N),
                conv(N, M, kernel_size=5, stride=2), # 16
                Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
            )
            self.post_decoder = nn.Sequential(
                Win_noShift_Attention(dim=M, num_heads=8, window_size=4, shift_size=2),
                deconv(M, N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                deconv(N, N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                Win_noShift_Attention(dim=N, num_heads=8, window_size=8, shift_size=4),
                deconv(N, N, kernel_size=5, stride=2),
                GDN(N, inverse=True),
                deconv(N, 3, kernel_size=5, stride=2),
            )



    def forward(self, x):
        if self.model_type == 0:
            out = self.res1(x)
            out = self.res2(out)
            out = self.attn1(out)
            out = self.res3(out) 
            out = self.res4(out) 

            out = self.res5(out) 
            out = self.attn2(out) 
            out = self.res6(out)
            out = self.res7(out)
            out = self.res8(out)  #ddd
            return out
        else: 
            out = x 
            out = self.post_encoder(out)
            out = self.post_decoder(out) 
            out += x 
            return out            




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
    

########################################################################
########################################################################
#################################################################

from compressai.models import MeanScaleHyperprior
from compressai.entropy_models import EntropyBottleneck,GaussianConditional


class MaskCompressor(MeanScaleHyperprior):
    def __init__(self, input_dim = 320, N = 320):
        super().__init__(N = N,M = N)

        self.input_dim = input_dim
        self.entropy_bottleneck = EntropyBottleneck(N)

        self.g_a = nn.Sequential(
            conv(input_dim, N),
            GDN(N),
            conv(N, N),
            GDN(N),
        )

        self.g_s = nn.Sequential(
            deconv(N, N),
            GDN(N, inverse=True),
            deconv(N, input_dim),
        )



        self.gaussian_conditional = GaussianConditional(None)


        self.h_a = nn.Sequential(
            conv(N, N),
            nn.LeakyReLU(inplace=True),
            conv(N, N),


        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.LeakyReLU(inplace=True),###
            deconv(N, N * 2),
        )

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        #print("lo shape di y è: ",y.shape,"  ",scales_hat.shape," ",means_hat.shape)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def print_information(self):


        print("**************************************************************************")
        model_tr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        model_fr_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad== False)
        print(" trainable parameters: ",model_tr_parameters)
        print(" freeze parameterss: ", model_fr_parameters)
        return model_tr_parameters

    def extract_map(self,x,net):

        
        out_p0 = net.forward_single_quality(x, 
                                                  quality = 0.0001, 
                                                  training = False, 
                                                  force_enhanced = True,
                                                  mask_pol = "point-based-std")
        out_dec_completo = net.forward_single_quality(x, 
                                                  quality = 10,
                                                  training = False,
                                                  force_enhanced = True, 
                                                  mask_pol = "point-based-std")
        
        


        y_hat_p0 = out_p0["y_hat"]
        y_hat_p0 = Variable(y_hat_p0, requires_grad=True)
        x_hat_p0 = net.g_s[1](y_hat_p0)

        y_hat_e = out_dec_completo["y_hat"]
        target = net.g_s[1](y_hat_e)



        loss = (255**2)*torch.mean((x_hat_p0 - target)**2) #dddd

        net.g_s[1].zero_grad()  
        loss.backward()


        gradient = y_hat_p0.grad
        gradient = gradient.to("cuda")
        difference_input = torch.abs(y_hat_p0 - y_hat_e)
        map = torch.abs(gradient)*difference_input

        map = self.forward(map)
        return  map