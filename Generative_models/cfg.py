# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import argparse


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--sw_type',
        type=str,
        help='The reload model path')
    parser.add_argument(
        '--max_epoch',
        type=int,
        default=200,
        help='number of epochs of training')
    parser.add_argument(
        '--L',
        type=int,
        default=1000,
        help='L')
    parser.add_argument(
        '--k',
        type=int,
        default=2,
        help='k')
    parser.add_argument(
        '--s_lr',
        type=float,
        default=0.01,
        help='s_lr')
    parser.add_argument(
        '--s_max_iter',
        type=int,
        default=100,
        help='set the max iteration number')
    parser.add_argument(
        '--max_iter',
        type=int,
        default=10000,
        help='set the max iteration number')
    parser.add_argument(
        '-gen_bs',
        '--gen_batch_size',
        type=int,
        default=64,
        help='size of the batches')
    parser.add_argument(
        '-dis_bs',
        '--dis_batch_size',
        type=int,
        default=64,
        help='size of the batches')
    parser.add_argument(
        '--g_lr',
        type=float,
        default=0.0002,
        help='adam: gen learning rate')
    parser.add_argument(
        '--d_lr',
        type=float,
        default=0.0002,
        help='adam: disc learning rate')
    
    parser.add_argument(
        '--mlp_lr',
        type=float,
        default=0.0002,
        help='adam: mlp learning rate')
    
    parser.add_argument(
        '--lr_decay',
        action='store_true',
        help='learning rate decay or not')
    parser.add_argument(
        '--beta1',
        type=float,
        default=0.0,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--beta2',
        type=float,
        default=0.9,
        help='adam: decay of first order momentum of gradient')
    parser.add_argument(
        '--num_workers',
        type=int,
        default=8,
        help='number of cpu threads to use during batch generation')
    parser.add_argument(
        '--latent_dim',
        type=int,
        default=128,
        help='dimensionality of the latent space')
    parser.add_argument(
        '--img_size',
        type=int,
        default=32,
        help='size of each image dimension')
    parser.add_argument(
        '--channels',
        type=int,
        default=3,
        help='number of image channels')
    parser.add_argument(
        '--n_critic',
        type=int,
        default=1,
        help='number of training steps for discriminator per iter')
    parser.add_argument(
        '--val_freq',
        type=int,
        default=20,
        help='interval between each validation')
    parser.add_argument(
        '--print_freq',
        type=int,
        default=50,
        help='interval between each verbose')
    parser.add_argument(
        '--load_path',
        type=str,
        help='The reload model path')
    parser.add_argument(
        '--exp_name',
        type=str,
        help='The name of exp')
    parser.add_argument(
        '--f_type',
        type=str,
        help='The name of exp')
    parser.add_argument(
        '--d_spectral_norm',
        type=str2bool,
        default=False,
        help='add spectral_norm on discriminator?')
    parser.add_argument(
        '--g_spectral_norm',
        type=str2bool,
        default=False,
        help='add spectral_norm on generator?')
    parser.add_argument(
        '--detach',
        type=str2bool,
        default=True,
        help='add spectral_norm on generator?')
    parser.add_argument(
        '--copy',
        type=str2bool,
        default=False,
        help='add spectral_norm on generator?')
    parser.add_argument(
        '--dataset',
        type=str,
        default='cifar10',
        help='dataset type')
    parser.add_argument(
        '--data_path',
        type=str,
        default='./data',
        help='The path of data set')
    parser.add_argument('--init_type', type=str, default='normal',
                        choices=['normal', 'orth', 'xavier_uniform', 'false'],
                        help='The init type')
    parser.add_argument('--gf_dim', type=int, default=64,
                        help='The base channel num of gen')
    parser.add_argument('--df_dim', type=int, default=64,
                        help='The base channel num of disc')
    parser.add_argument(
        '--model',
        type=str,
        default='sngan_cifar10',
        help='path of model')
    parser.add_argument('--eval_batch_size', type=int, default=100)
    #parser.add_argument('--num_eval_imgs', type=int, default=200)
    parser.add_argument('--num_eval_imgs', type=int, default=50000)
    parser.add_argument(
        '--bottom_width',
        type=int,
        default=4,
        help="the base resolution of the GAN")
    parser.add_argument('--random_seed', type=int, default=12345)
    parser.add_argument('--nlines', type=int, default=5)
    parser.add_argument('--range_root_1', type=float, default=-1.0)
    parser.add_argument('--range_root_2', type=float, default=1.0)
    parser.add_argument('--range_source_1', type=float, default=-0.1)
    parser.add_argument('--range_source_2', type=float, default=0.1)
    parser.add_argument('--p', type=float, default=1.0)
    parser.add_argument('--fixed_trees', type=str2bool, default=False)
    parser.add_argument('--d', type=int, default=8193)
    parser.add_argument('--type_lines', type=str, default='sequence_of_lines')
    parser.add_argument('--mass_division', type=str, default='uniform')
    parser.add_argument('--hidden_dim_learnable_mass', type=int, default=64)
    parser.add_argument('--dropout_prob', type=float, default=0.2)
    parser.add_argument('--device', type=str, default='cuda')

    opt = parser.parse_args()
    return opt
