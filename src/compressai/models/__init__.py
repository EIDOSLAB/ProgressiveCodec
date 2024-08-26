# Copyright 2020 InterDigital Communications, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from .stf import SymmetricalTransFormer
from .cnn import WACNN

from .CHProg_cnn import ChannelProgresssiveWACNN
from .CHProgREM import PostRateProcessedNetwork


models = {
    'stf': SymmetricalTransFormer,
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
    


    if args.model == "restcm":
        net = models[args.model](N = args.N,
                                M = args.M,
                                multiple_decoder = args.multiple_decoder,
                                multiple_encoder = args.multiple_encoder,
                                dim_chunk = args.dim_chunk,
                                division_dimension = args.division_dimension,
                                lmbda_list = lmbda_list,
                                mask_policy = args.mask_policy,
                                joiner_policy = args.joiner_policy,
                                support_progressive_slices =args.support_progressive_slices,
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
