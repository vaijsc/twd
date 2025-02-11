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
        '--num_iter',
        type=int,
        default=2500,
        help='number of epochs of training')
    parser.add_argument(
        '--L',
        type=int,
        default=1000,
        help='L')
    parser.add_argument(
        '--n_lines',
        type=int,
        default=4,
        help='Number of lines in each tree')

    parser.add_argument(
        '--lr_sw',
        type=float,
        default=1e-3,
        help='learning rate of SW')
    parser.add_argument(
        '--lr_tsw_sl',
        type=float,
        default=1e-3,
        help='learning rate of TSW-SL')
    parser.add_argument(
        '--delta',
        type=float,
        default=50.,
        help='delta to tune distance-based')
    parser.add_argument(
        '--p',
        type=int,
        default=1,
        help='p')
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='25gaussians',
        help='Name of the dataset')   

    parser.add_argument(
        '--std',
        type=float,
        default=0.1,
        help='std to generate root of trees')
    parser.add_argument(
        '--num_seeds',
        type=int,
        default=0.1,
        help='std to generate root of trees')





    opt = parser.parse_args()
    return opt
