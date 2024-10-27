
from .cnn import WACNN

from .CHProg_cnn import ChannelProgresssiveWACNN
from .CHProgREM import PostRateProcessedNetwork


models = {

    'cnn': WACNN,
    "rate":PostRateProcessedNetwork,
    "channel":ChannelProgresssiveWACNN,

}

def get_model(args,device, lmbda_list):

    if args.model == "multidec":
        net = models[args.model]( N = args.N,
                                M = args.M,
                                multiple_decoder = args.multiple_decoder,
                                multiple_encoder = args.multiple_encoder,
                                multiple_hyperprior = args.multiple_hyperprior,
                                dim_chunk = args.dim_chunk,
                                division_dimension = args.division_dimension,
                                lmbda_list = lmbda_list,
                                mask_policy = args.mask_policy,
                                joiner_policy = args.joiner_policy,
                                support_progressive_slices =args.support_progressive_slices,
                                double_dim = args.double_dim,
                                delta_encode = args.delta_encode,
                                residual_before_lrp = args.residual_before_lrp,
                                num_adapter = args.num_adapter,
                                milestones = args.milestones
                        ) 
    


    elif  args.model == "progressive":
        net = models[args.model]( N = args.N,
                                M = args.M,
                                multiple_decoder = args.multiple_decoder,
                                dim_chunk = args.dim_chunk,
                                division_dimension = args.division_dimension,
                                lmbda_list = lmbda_list

                        ) 
      
    elif args.model == "channel":
        net = models[args.model]( N = args.N,
                                M = args.M,
                                multiple_decoder = args.multiple_decoder,
                                multiple_encoder = args.multiple_encoder,
                                multiple_hyperprior = args.multiple_hyperprior,
                                dim_chunk = args.dim_chunk,
                                division_dimension = args.division_dimension,
                                lmbda_list = lmbda_list,
                                mask_policy = args.mask_policy,
                                joiner_policy = args.joiner_policy,
                                support_progressive_slices =args.support_progressive_slices,
                                double_dim = args.double_dim,
                                delta_encode = args.delta_encode,
                                residual_before_lrp = args.residual_before_lrp,
                                support_std = args.support_std,
                                total_mu_rep = args.total_mu_rep,
                                all_scalable = args.all_scalable,
                                u_net_post = args.u_net_post
                        )  
    elif args.model == "progressive_res":
        net = models[args.model]( N = args.N,
                                M = args.M,
                                multiple_decoder = args.multiple_decoder, #ddd
                                dim_chunk = args.dim_chunk,
                                division_dimension = args.division_dimension,
                                lmbda_list = lmbda_list,
                                shared_entropy_estimation = args.shared_entropy_estimation,
                                joiner_policy = args.joiner_policy

                        )       
       
    else:
        net = models[args.model]( N = args.N,
                                M = args.M,
                                )


    net = net.to(device)
    return net
