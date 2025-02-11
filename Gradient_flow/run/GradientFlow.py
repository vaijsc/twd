import os
os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

import numpy as np
import torch
from torch import optim
import ot
import pandas as pd
from tqdm import tqdm
from IPython import display
import time
import timeit
import matplotlib.pyplot as pl

import os
import sys
sys.path.append('../code/')

from utils_GF import load_data,w2
from SWGG import SWGG_smooth,get_SWGG_smooth
import gradient_flow
import TW

dataset_name = 'swiss_roll'
seed=0
np.random.seed(seed)
N = 100  # Number of samples from p_X
X = load_data(name=dataset_name, n_samples=N,dim=2)
X -= X.mean(dim=0)[np.newaxis,:]  # Normalization
#X-=torch.tensor([5,5])
meanX = 0
# Show the dataset
_, d = X.shape
print(X.shape)
fig = pl.figure(figsize=(5,5))
pl.scatter(X[:,0], X[:,1])
pl.show()

results_folder = './Results/Gradient_Flow'
if not os.path.isdir(results_folder):
    os.mkdir(results_folder)
foldername = os.path.join(results_folder, 'Gifs')
if not os.path.isdir(foldername):
    os.mkdir(foldername)
    
foldername = os.path.join(results_folder, 'Gifs', dataset_name + '_Comparison')
if not os.path.isdir(foldername):
    os.mkdir(foldername)

# Use GPU if available, CPU otherwise
print(torch.cuda.is_available())
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# Number of iterations for the optimization process
nofiterations = 2500

modes = ['linear', 'linear', 'linear','linear','linear', 'linear']
titles = ['SW', 'maxSW','LCVSW','SWGG_random','MaxTSW', 'TSW-SL']

lear_rates=6*[1e-1]

nb_iteration = [1,50,1,1,50,1]
n_proj = [100,1,100,100,1,100]


# Define the initial distribution
temp = np.random.normal(loc=meanX, scale=.25, size=(N,d))

# Define the variables to store the loss (2-Wasserstein distance) for each defining function and each problem
dist='w2'
w2_dist = np.nan * np.zeros((nofiterations, len(modes)))


# Define the optimizers
Y = list()
optimizer = list()
gsw_res = list()
for k in range(len(modes)):
    Y.append(torch.tensor(temp, dtype=torch.float, device=device, requires_grad=True))
    optimizer.append(optim.Adam([Y[k]], lr = lear_rates[k]))
    gsw_res.append(gradient_flow.GF(ftype=modes[k], nofprojections=n_proj[k],device=device))


s=len(modes)


fig = pl.figure(figsize=(4*s, 8+3))
# grid = pl.GridSpec(3, s, wspace=.4, hspace=0.3)
plot_fig = True
mlp = mlp.MLP(X.shape[1], 64, 3, 0.2).to('cuda')
tree = gradient_flow.Tree(34, X.shape[1], -1.0, 1.0, -0.1, 0.1, 3, 'cuda', 'sequence_of_lines', 'True')
theta_twd, intercept_twd, subsequent_sources_twd = tree.get_tree()

for i in range(nofiterations):
    print(i,end=' ')
    loss = list()
    theta = torch.ones(len(modes),d)
    
    #X = load_data(name=dataset_name, n_samples=N,dim=2)
    #X -= X.mean(dim=0)[np.newaxis,:]  # Normalization
    
    for k in range(s):
        # Loss computation
        loss_ = 0
        if k==0:
            loss_ += gsw_res[k].sw(X.to(device), Y[k],theta=None)
        
        if k==1:
            #print(X,Y[k])
            l,theta[k],loss_max=gsw_res[k].max_sw(X.to(device),Y[k],iterations=nb_iteration[k],lr=lear_rates[k])
            loss_ +=l
            
        if k==2:
            loss_ += gradient_flow.LCVSW(X.to(device), Y[k].to(device),theta=None)
            
        if k==3:
            l,theta[k] = gsw_res[k].SWGG_CP(X.to(device), Y[k].to(device),theta=None)
            loss_+=l
        if k==4:
            loss_ += gradient_flow.maxTWD(X.to(device), Y[k],n_lines= 5)
        if k==5:
            loss_ += gradient_flow.TWD(X.to(device), Y[k],theta_twd, intercept_twd, subsequent_sources_twd, mlp)

            
        # Optimization step
        loss.append(loss_)
        optimizer[k].zero_grad()
        loss[k].backward()
        optimizer[k].step()
        
        # Compute the 2-Wasserstein distance to compare the distributions
        if dist=='w2':
            w2_dist[i, k] = w2(X.detach().cpu().numpy(), Y[k].detach().cpu().numpy())

 
             
        theta2=theta.numpy()   
        if plot_fig:
            if k==0 or k==4:
                temp = Y[k].detach().cpu().numpy()
                #pl.subplot(grid[k//s, k%s])
                pl.scatter(X[:,0], X[:,1])
                pl.scatter(temp[:,0], temp[:,1],c='r')
                pl.title(titles[k], fontsize=10)
                xlim1,xlim2=pl.xlim()
                ylim1,ylim2=pl.ylim()
            else :
                temp = Y[k].detach().cpu().numpy()
                #pl.subplot(grid[k//s, k%s])
                pl.scatter(X[:,0], X[:,1])
                pl.scatter(temp[:,0], temp[:,1],c='r')
                pl.axline((0,0), theta2[k][0:2], color='C2')
                pl.xlim(xlim1,xlim2)
                pl.ylim(ylim1,ylim2)
                pl.title(titles[k], fontsize=10)
    #print(theta[k])
    if plot_fig:
    # Plot the 2-Wasserstein distance
         # pl.subplot(grid[1, 0:s])
        pl.plot(np.log10(w2_dist[:,:]), linewidth=3)
        pl.title('2-Wasserstein Distance', fontsize=10)
        pl.ylabel(r'$Log_{10}(W_2)$', fontsize=22)
        pl.legend(titles, fontsize=10, loc='lower left')
        
        
        display.clear_output(wait=True)
        display.display(pl.gcf()) 
        time.sleep(1e-5)

        # Save the figure 
        fig.savefig(foldername + '/img%03d.png'%(i))
        for k in range(s):
             # pl.subplot(grid[:, k])
            pl.cla()

    if i % 100 == 0: 
        np.savetxt("Results/Gradient_Flow/"+dataset_name+"_maxTW.txt"+str(i),w2_dist[:,0])

    
            