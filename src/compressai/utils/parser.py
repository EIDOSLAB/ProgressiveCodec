import argparse
from compressai.models import models




def parse_args(argv):
    parser = argparse.ArgumentParser(description="Example training script.") #dddd

    parser.add_argument("--checkpoint", type=str, default = "none") #dddd
    parser.add_argument("--name_model",type=str,default = "none")


    parser.add_argument("-cl", "--cluster", type=str, default = "hssh",choices=["hssh","nautilus"], help="Training dataset")
    parser.add_argument("-m","--model",default="channel",choices=models.keys(),help="Model architecture (default: %(default)s)",)
    
    parser.add_argument("-e","--epochs",default=150,type=int,help="Number of epochs (default: %(default)s)",)
    parser.add_argument( "-lr", "--learning-rate", default=1e-4, type=float, help="Learning rate (default: %(default)s)",)
    parser.add_argument("-n","--num-workers",type=int,default=8,help="Dataloaders threads (default: %(default)s)",) #u_net_post
    #dddd
    parser.add_argument("--u_net_post",type=int,default=2,help="Dataloaders threads (default: %(default)s)",)
    parser.add_argument("--lambda_list",dest="lmbda_list", nargs='+', type=float, default = [ 0.0055,0.04])
    parser.add_argument("--division_dimension", nargs='+', type=int, default = [320, 640])
    parser.add_argument("--inner_dimensions", nargs='+', type=int, default = [192, 192]) #ddddfffff
    parser.add_argument("--list_quality", nargs='+', type=int, default = [0])
    parser.add_argument( "--batch_size", type=int, default=16, help="batch_size")
    parser.add_argument( "--dim_chunk", type=int, default=32, help="dim chunk")

    parser.add_argument("--support_std", action="store_true", help="KD base")
    parser.add_argument("--all_scalable", action="store_true", help="KD base")
    parser.add_argument("--total_mu_rep", action="store_true", help="KD base")
    parser.add_argument("--kd_base", action="store_true", help="KD base")
    parser.add_argument("--freeze_base", action="store_true", help="KD base")#dddd
    parser.add_argument("--residual_before_lrp", action="store_true", help="KD base")


    parser.add_argument("--num_images", type=int, default=300000, help="num images") #ddddddd

    parser.add_argument("--N", type=int, default=192, help="N")#ddddd#ddd
    parser.add_argument("--M", type=int, default=640, help="M")
    parser.add_argument("--patience", type=int, default=7, help="patience")#ddddddd

    parser.add_argument("--code", type=str, default = "0001", help="Batch size (default: %(default)s)")
    parser.add_argument("--num_images_val", type=int, default=816, help="Batch size (default: %(default)s)")
    parser.add_argument("--mask_policy", type=str, default = "two-levels", help="mask_policy")
    parser.add_argument("--valid_batch_size",type=int,default=16,help="Test batch size (default: %(default)s)",)
    parser.add_argument("--test_batch_size", type=int, default=1, help="Test batch size (default: %(default)s)", )
    parser.add_argument("--aux-learning-rate", default=1e-3, type=float, help="Auxiliary loss learning rate (default: %(default)s)",)
    parser.add_argument("--patch-size",type=int,nargs=2,default=(256, 256),help="Size of the patches to be cropped (default: %(default)s)",)
    parser.add_argument("--cuda", action="store_true", help="Use cuda")
    parser.add_argument("--freeze", action="store_true", help="Use cuda") #dddfff
    parser.add_argument("--save", action="store_true", default=True, help="Save model to disk")
    parser.add_argument("--multiple_decoder", action="store_true", help="Use cuda")
    parser.add_argument("--multiple_encoder", action="store_true", help="Use cuda")
    parser.add_argument("--multiple_hyperprior", action="store_true", help="Use cuda")
    parser.add_argument("--double_dim", action="store_true", help="Use cuda")
    parser.add_argument("--mutual", action="store_true", help="Use cuda")
    parser.add_argument("--delta_encode", action="store_true", help="delta encoder")



    parser.add_argument("--milestones", nargs='+', type=float, default = [0,0.75,1.5,3,10])


    parser.add_argument("-see","--shared_entropy_estimation", action="store_true", help="Use cuda")
    parser.add_argument("--continue_training", action="store_true", help="continue training of the checkpoint")

    parser.add_argument("--seed", type=float, help="Set random seed for reproducibility")
    parser.add_argument("--sampling_training", action="store_true", help="Save model to disk")
    parser.add_argument("--joiner_policy", type=str, default = "res",help="Path to a checkpoint") 
    parser.add_argument("--clip_max_norm",default=1.0,type=float,help="gradient clipping max norm (default: %(default)s",)

    parser.add_argument("--checkpoint_base", type=str, default =  "none",help="Path to a checkpoint") #"/scratch/base_devil/weights/q2/model.pth"
    parser.add_argument("--checkpoint_enh", type=str, default =  "none",help="Path to a checkpoint") #"/scratch/base_devil/weights/q5/model.pth"
    parser.add_argument("--tester", action="store_true", help="use common lrp for progressive") #dddddd
    parser.add_argument("--support_progressive_slices",default=4,type=int,help="support_progressive_slices",) #ssss
    args = parser.parse_args(argv)
    return args

