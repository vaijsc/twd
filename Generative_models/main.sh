GPU=0
seed=1
L=10
CUDA_VISIBLE_DEVICES=${GPU} taskset --cpu-list ${GPU}0-${GPU}9 python3 trainsw.py -gen_bs 128 -dis_bs 128 --dataset celeba --img_size 64 --max_iter 50000 --model sngan_celeba --latent_dim 128 --gf_dim 256 --df_dim 128 --g_spectral_norm False --d_spectral_norm True --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --n_critic 5 --val_freq 1 --L ${L} --sw_type lcvsw --random_seed ${seed} --exp_name SW_L${L}_seed${seed} &
CUDA_VISIBLE_DEVICES=${GPU} taskset --cpu-list ${GPU}0-${GPU}9 python3 trainsw.py -gen_bs 128 -dis_bs 128 --dataset celeba --img_size 64 --max_iter 50000 --model sngan_celeba --latent_dim 128 --gf_dim 256 --df_dim 128 --g_spectral_norm False --d_spectral_norm True --g_lr 0.0002 --d_lr 0.0002 --beta1 0.0 --beta2 0.9 --init_type xavier_uniform --n_critic 5 --val_freq 1 --L ${L} --sw_type tw --random_seed ${seed} --fixed_trees True --mass_division learnable --type_lines concurrent_lines --nlines 6 --exp_name TW_L${L}_seed${seed}
