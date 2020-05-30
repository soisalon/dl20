#!/bin/bash
#SBATCH -J dl_train
#SBATCH --export=USERAPPL,WRKDIR,LD_LIBRARY_PATH
#SBATCH --chdir=/wrk/users/eliel/projects/dl_course20/dl20
#SBATCH -o /wrk/users/eliel/projects/dl_course20/jobs/res/%J.txt
#SBATCH -e /wrk/users/eliel/projects/dl_course20/jobs/res/%J.txt
#SBATCH -t 2-0
#SBATCH --mem 10G
#SBATCH -c 10
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=eliel.soisalon-soininen@helsinki.fi

# interactive
# srun -t 10:00:00 --mem=10G -p gpu-short --gres=gpu:1 -c 10 --pty bash


module purge
module load Python/3.7.0-intel-2018b
# which version of cuda to install? 10.2
module load CUDA/10.1.105

### How to use GPU - download CUDA module ?


echo "training cnn for DL"

# --array=0-5
#ID=SLURM_ARRAY_TASK_ID


# train model
srun $USERAPPL/ve37/bin/python3 src/train.py \
    --dev_ratio 0.1 \
    --seed 100 \
    --cv_folds 10 \
    --emb_pars enc=elmo_2x1024_128_2048cnn_1xhighway dim=2 \
    --n_epochs 10 \
    --batch_size 32 \
    --loss_fn bce \
    --optim adadelta \
    --opt_params lr=1.0 rho=0.95 eps=1e-6 \
    --model_name BaseCNN \
    --n_conv_layers 1 \
    --kernel_shapes 256x2 \
    --strides 1x1 \
    --input_shape 256x50 \
    --n_kernels 10 \
    --conv_act_fn relu \
    --h_units 64 \
    --fc_act_fn relu \
    --out_act_fn sigmoid \
    --dropout 0.5




