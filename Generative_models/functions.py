# -*- coding: utf-8 -*-
# @Date    : 2019-07-25
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

import os
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from imageio import imsave
from tqdm import tqdm
from copy import deepcopy
import logging
from utils.inception_score import get_inception_score
from utils.fid_score import calculate_fid_given_paths
from TW import TW, generate_trees_frames
import cfg
from torch import optim

logger = logging.getLogger(__name__)

args = cfg.parse_args()


class Tree():
    def __init__(self, L, d, range_root_1, range_root_2, range_source_1, range_source_2, nlines, device, type_lines, fixed_trees):
        self.L = L
        self.d = d
        self.range_root_1 = range_root_1
        self.range_root_2 = range_root_2
        self.range_source_1 = range_source_1
        self.range_source_2 = range_source_2
        self.nlines = nlines
        self.device = device
        self.type_lines = type_lines
        self.fixed_trees = fixed_trees
        if self.fixed_trees:
            self.already_generate = 0

    def get_tree(self):
        if self.fixed_trees:
            if self.already_generate:
                return self.theta, self.intercept, self.subsequent_sources
            else:
                self.already_generate = 1
                self.generate_trees()
                return self.theta, self.intercept, self.subsequent_sources
        else:
            self.generate_trees()
            return self.theta, self.intercept, self.subsequent_sources

    def generate_trees(self):
        self.theta, self.intercept, self.subsequent_sources = generate_trees_frames(self.L, self.d, self.range_root_1, self.range_root_2, self.range_source_1, self.range_source_2, self.nlines, self.device, self.type_lines)
def train(args, gen_net: nn.Module, dis_net: nn.Module, mlp: nn.Module, gen_optimizer, dis_optimizer, mlp_optimizer, gen_avg_param, train_loader, epoch,
          writer_dict, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()
    mlp = mlp.train()
    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        true_bs = imgs.shape[0]
        if (true_bs < args.gen_batch_size):
            break
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)

        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (imgs.shape[0], args.latent_dim)))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)

        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        d_loss.backward()
        dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), global_steps)

        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            gen_optimizer.zero_grad()
            mlp_optimizer.zero_grad()
            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.gen_batch_size, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            fake_validity = dis_net(gen_imgs)

            # cal loss
            g_loss = -torch.mean(fake_validity)
            g_loss.backward()
            gen_optimizer.step()
            mlp_optimizer.step()

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1
        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1

def train_sw(args, gen_net: nn.Module, dis_net: nn.Module, mlp: nn.Module, gen_optimizer, dis_optimizer, mlp_optimizer, gen_avg_param, train_loader, epoch,
          writer_dict, schedulers=None):
    global theta, intercept, subsequent_sources
    writer = writer_dict['writer']
    gen_step = 0

    # train mode
    gen_net = gen_net.train()
    dis_net = dis_net.train()
    mlp = mlp.train()

    for iter_idx, (imgs, _) in enumerate(tqdm(train_loader)):
        true_bs = imgs.shape[0]
        if (true_bs < args.gen_batch_size):
            break
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = imgs.type(torch.cuda.FloatTensor)
        true_bs = imgs.shape[0]
        # Sample noise as generator input
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (true_bs, args.latent_dim)))

        # ---------------------
        #  Train Discriminator
        # ---------------------
        dis_optimizer.zero_grad()

        real_validity = dis_net(real_imgs)
        fake_imgs = gen_net(z).detach()
        assert fake_imgs.size() == real_imgs.size()

        fake_validity = dis_net(fake_imgs)
        # cal loss
        d_loss = torch.mean(nn.ReLU(inplace=True)(1.0 - real_validity)) + \
                 torch.mean(nn.ReLU(inplace=True)(1 + fake_validity))
        d_loss.backward()
        dis_optimizer.step()
        writer.add_scalar('d_loss', d_loss.item(), global_steps)
        # -----------------
        #  Train Generator
        # -----------------
        if global_steps % args.n_critic == 0:
            real_validity, real_features = dis_net(real_imgs, return_feature=True)
            gen_optimizer.zero_grad()
            mlp_optimizer.zero_grad()
            gen_z = torch.cuda.FloatTensor(np.random.normal(0, 1, (true_bs, args.latent_dim)))
            gen_imgs = gen_net(gen_z)
            fake_validity, fake_features = dis_net(gen_imgs, return_feature=True)
            # cal loss
            X = torch.cat([real_features[-1].view(true_bs, -1),real_validity.view(true_bs,-1)],dim=1)
            Y = torch.cat([fake_features[-1].view(true_bs, -1),fake_validity.view(true_bs,-1)],dim=1)
            tree = Tree(args.L, args.d, args.range_root_1, args.range_root_2, args.range_source_1, args.range_source_2, args.nlines, args.device, args.type_lines, args.fixed_trees)
            theta, intercept, subsequent_sources = tree.get_tree()
            
            # cal loss
            if(args.sw_type=='sw'):
                g_loss =  SW(X,Y,L=args.L)
            elif(args.sw_type=='lcvsw'):
                g_loss =  LCVSW(X,Y,L=args.L)
            elif(args.sw_type=='ucvsw'):
                g_loss =  UCVSW(X,Y,L=args.L)
            elif(args.sw_type=='tw'):
                g_loss =  TWD(X,Y,theta,intercept,subsequent_sources, mlp, args.mass_division, args.p, args.device)

            elif(args.sw_type=='max_sw'):
                g_loss =  max_sw(X,Y)

            g_loss.backward()
            gen_optimizer.step()
            mlp_optimizer.step()
            #print("Check_gradient: ", mlp.linear1.weight.grad)

            # adjust learning rate
            if schedulers:
                gen_scheduler, dis_scheduler = schedulers
                g_lr = gen_scheduler.step(global_steps)
                d_lr = dis_scheduler.step(global_steps)
                writer.add_scalar('LR/g_lr', g_lr, global_steps)
                writer.add_scalar('LR/d_lr', d_lr, global_steps)

            # moving average weight
            for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
                avg_p.mul_(0.999).add_(0.001, p.data)

            writer.add_scalar('g_loss', g_loss.item(), global_steps)
            gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item()))

        writer_dict['train_global_steps'] = global_steps + 1


def validate(args, fixed_z, fid_stat, gen_net: nn.Module, writer_dict):
    writer = writer_dict['writer']
    global_steps = writer_dict['valid_global_steps']

    # eval mode
    gen_net = gen_net.eval()

    # generate images
    sample_imgs = gen_net(fixed_z)
    img_grid = make_grid(sample_imgs, nrow=5, normalize=True, scale_each=True)

    # get fid and inception score
    fid_buffer_dir = os.path.join(args.path_helper['sample_path'], 'fid_buffer')
    os.makedirs(fid_buffer_dir)

    eval_iter = args.num_eval_imgs // args.eval_batch_size
    img_list = list()
    for iter_idx in tqdm(range(eval_iter), desc='sample images'):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

        # Generate a batch of images
        gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu', torch.uint8).numpy()
        for img_idx, img in enumerate(gen_imgs):
            file_name = os.path.join(fid_buffer_dir, f'iter{iter_idx}_b{img_idx}.png')
            imsave(file_name, img)
            #exit()
        img_list.extend(list(gen_imgs))

    # get inception score
    logger.info('=> calculate inception score')
    if args.dataset.lower() == 'lsun_church' or args.dataset.lower() == 'celebahq' :
        mean, std = get_inception_score(img_list,bs=16)
    elif args.dataset.lower() == 'stl10':
        mean, std = get_inception_score(img_list,bs=32)
    else:
        mean, std = get_inception_score(img_list,bs=100)

    # get fid score
    logger.info('=> calculate fid score')
    if args.dataset.lower() == 'lsun_church' or args.dataset.lower() == 'celebahq' :
        fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None,batch_size=args.eval_batch_size)
    elif args.dataset.lower() == 'stl10':
        fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None,batch_size=args.eval_batch_size)
    else:
        fid_score = calculate_fid_given_paths([fid_buffer_dir, fid_stat], inception_path=None,batch_size=args.eval_batch_size)
    #os.system('rm -r {}'.format(fid_buffer_dir))

    writer.add_image('sampled_images', img_grid, global_steps)
    writer.add_scalar('Inception_score/mean', mean, global_steps)
    writer.add_scalar('Inception_score/std', std, global_steps)
    writer.add_scalar('FID_score', fid_score, global_steps)

    writer_dict['valid_global_steps'] = global_steps + 1

    return mean, fid_score


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)


def copy_params(model):
    flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten


def rand_projections(dim, num_projections=1000,device='cpu'):
    projections = torch.randn((num_projections, dim),device=device)
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections


def one_dimensional_Wasserstein_prod(X,Y,theta,p):
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    X_prod = X_prod.view(X_prod.shape[0], -1)
    Y_prod = Y_prod.view(Y_prod.shape[0], -1)
    wasserstein_distance = torch.abs(
        (
                torch.sort(X_prod, dim=0)[0]
                - torch.sort(Y_prod, dim=0)[0]
        )
    )
    wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=0,keepdim=True)
    return wasserstein_distance
#generate_trees_frames(L, d, range_root_1 = -1.0, range_root_2 = 1.0, range_source_1 = -0.1, range_source_2 = 0.1, nlines = 5, device = 'cuda')

def SW(X, Y, theta = None, L=10, p=2,device="cuda"):
    dim = X.size(1)
    if theta is None:
        theta = rand_projections(dim, L, device)
    sw=one_dimensional_Wasserstein_prod(X,Y,theta,p=p).mean()
    return  torch.pow(sw,1./p)

def TWD(X, Y, theta, intercept, subsequent_sources, mlp, mass_division = 'learnable', p = 1, device="cuda"):
    L = theta.shape[0]
    nlines = theta.shape[1]
    treewasserstein = TW(mlp = mlp, nofprojections = L, nlines = nlines, p = p, mass_division = mass_division, device = device)
    return treewasserstein.tw(X, Y, theta, intercept, subsequent_sources)

def max_sw(X,Y,iterations=5,lr=1e-4, device="cuda"):
    N,dn = X.shape
    M,dm = Y.shape
    assert dn==dm and M==N
#         if self.theta is None:
  
    theta=torch.randn((1,dn),device=device,requires_grad=True)
    theta.data/=torch.sqrt(torch.sum((theta.data)**2))
    optimizer=optim.Adam([theta],lr=lr)
    loss_l=[]
    for i in range(iterations):
        optimizer.zero_grad()
        loss=-SW(X.to(device),Y.to(device), theta.to(device))
        #print('test4')
        loss_l.append(loss.data)
        loss.backward(retain_graph=True)
        optimizer.step()
        theta.data/=torch.norm(theta.data)
        #print('test5')

    res = SW(X.to(device),Y.to(device),theta.to(device))
    return res

def LCVSW(X,Y,L=10,p=2,device="cuda"):
    dim = X.size(1)
    m_1 = torch.mean(X,dim=0)
    m_2 = torch.mean(Y,dim=0)
    diff_m1_m2= m_1-m_2
    G_mean = torch.mean((diff_m1_m2)**2) #+ (sigma_1-sigma_2)**2
    theta = rand_projections(dim, L, device)
    hat_G = torch.sum(theta*(diff_m1_m2),dim=1)**2 #+(sigma_1-sigma_2)**2
    diff_hat_G_mean_G = hat_G - G_mean
    hat_sigma_G_square = torch.mean((diff_hat_G_mean_G)**2)
    distances = one_dimensional_Wasserstein_prod(X,Y,theta,p=p)
    hat_A = distances.mean()
    hat_C_G = torch.mean((distances-hat_A)*(diff_hat_G_mean_G))
    hat_alpha = hat_C_G/(hat_sigma_G_square+1e-24)
    Z = hat_A - hat_alpha*torch.mean(diff_hat_G_mean_G)
    return torch.pow(torch.mean(Z),1./p)

def UCVSW(X,Y,L=10,p=2,device="cuda"):
    dim = X.size(1)
    m_1 = torch.mean(X,dim=0)
    m_2 = torch.mean(Y,dim=0)
    diff_m1_m2= m_1-m_2
    diff_X_m1 = X-m_1
    diff_Y_m2 = Y-m_2
    G_mean = torch.mean((diff_m1_m2)**2) +  torch.mean((diff_X_m1)**2)+  torch.mean((diff_Y_m2)**2)
    theta = rand_projections(dim, L, device)
    hat_G = torch.sum(theta*(diff_m1_m2),dim=1)**2 +torch.mean(torch.matmul(theta,diff_X_m1.transpose(0,1))**2,dim=1)+torch.mean(torch.matmul(theta,diff_Y_m2.transpose(0,1))**2,dim=1)
    diff_hat_G_mean_G = hat_G - G_mean
    hat_sigma_G_square = torch.mean((diff_hat_G_mean_G)**2)
    distances = one_dimensional_Wasserstein_prod(X,Y,theta,p=p)
    hat_A = distances.mean()
    hat_C_G = torch.mean((distances-hat_A)*(diff_hat_G_mean_G))
    hat_alpha = hat_C_G/(hat_sigma_G_square+1e-24)
    Z = hat_A - hat_alpha*torch.mean(diff_hat_G_mean_G)
    return torch.pow(torch.mean(Z),1./p)
