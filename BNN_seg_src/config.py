import argparse
import os
from distutils.util import strtobool


parser = argparse.ArgumentParser()

# Environment
parser.add_argument("--is_train", type=strtobool, default='true')
parser.add_argument("--tensorboard", type=strtobool, default='true')
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument("--num_gpu", type=int, default=1)
parser.add_argument("--num_work", type=int, default=6)
parser.add_argument("--exp_dir", type=str, default="../BNN_seg_exp")
parser.add_argument("--exp_load", type=str, default= None)

# Data
parser.add_argument("--data_dir", type=str, default="../data") # change to 
parser.add_argument("--data_name", type=str, default="AMZ") # CamVid
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--rgb_range', type=int, default=1)



# Model
parser.add_argument('--uncertainty', default='combined',
                    choices=('normal', 'epistemic', 'aleatoric', 'combined'))
parser.add_argument('--in_channels', type=int, default=3)
parser.add_argument('--n_classes', type=int, default=5) # 12 for Camvid, 5 for AMZ
parser.add_argument('--n_feats', type=int, default=32)
parser.add_argument('--var_weight', type=float, default=1.)
parser.add_argument('--drop_rate', type=float, default=0.2)

# Train
parser.add_argument("--epochs", type=int, default=400)
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--decay", type=str, default='100-200-300-350-375') #'50-100-150-200' # 2-4-6-8-10-12-14-16-18-20' # '40-80-120-160-200'
parser.add_argument("--gamma", type=float, default=0.8)
parser.add_argument("--optimizer", type=str, default='adam',
                    choices=('sgd', 'adam', 'rmsprop'))
parser.add_argument("--weight_decay", type=float, default=1e-4)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--betas", type=tuple, default=(0.9, 0.999))
parser.add_argument("--epsilon", type=float, default=1e-8)

# Test
parser.add_argument('--n_samples', type=int, default=25)
parser.add_argument('--test_interval', type=int, default=20)

def save_args(obj, defaults, kwargs):
    for k,v in defaults.iteritems():
        if k in kwargs: v = kwargs[k]
        setattr(obj, k, v)

### get config for training
def get_config():
    config = parser.parse_args()
    config.data_dir = os.path.expanduser(config.data_dir)
    return config





