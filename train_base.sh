#!/bin/bash
#SBATCH -J dl_tr_seq
#SBATCH --export=USERAPPL,WRKDIR,LD_LIBRARY_PATH
#SBATCH --chdir=/wrk/users/eliel/projects/dl_course20/
#SBATCH -o /wrk/users/eliel/projects/dl_course20/jobs/res/%A_%a.txt
#SBATCH -e /wrk/users/eliel/projects/dl_course20/jobs/err/%A_%a.txt
#SBATCH -t 0-5:00:00
#SBATCH -c 10
#SBATCH --mail-type=END
#SBATCH --mail-user=eliel.soisalon-soininen@helsinki.fi
#SBATCH --mem=50G
#SBATCH --gres=gpu:1
#SBATCH -p gpu-short


# interactive
# srun -t 10:00:00 --mem=50G -p gpu-short --gres=gpu:1 -c 10 --pty bash


module purge
module load Python/3.7.0-intel-2018b
module load CUDA/10.1.105


echo "training cnn for DL"

ID=SLURM_ARRAY_TASK_ID
# ID=2

EMBS=(enc=word2vec)
# KS=(300x2 300x3 300x4 300x5 300x6 300x8 300x10 300x12 300x14)
KS=(300x5)
# N_KS=(1 10 50 100 200 300 400 500 600 700 800 900)
N_KS=(100)
NC=(1)
MODS=(BaseCNN)
INS=(300x100)
BS=(64)
OPTS=(adadelta)
# HS=(10 50 100 200 500)
HS=(100)
DROPS=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
# train model
# srun -n 4 --exclusive $USERAPPL/ve37/bin/python3 dl20/src/train.py \
srun $USERAPPL/ve37/bin/python3 dl20/src/train.py \
    --dev_ratio 0.1 \
    --seed 100 \
    --use_seqs 1 \
    --emb_pars ${EMBS[$ID % ${#EMBS[@]}]} \
    --n_epochs 30 \
    --batch_size ${BS[$ID % ${#BS[@]}]} \
    --loss_fn bce \
    --optim ${OPTS[$ID % ${#OPTS[@]}]}\
    --opt_params default \
    --model_name ${MODS[$ID % ${#MODS[@]}]}\
    --n_conv_layers ${NC[$ID % ${#NC[@]}]} \
    --kernel_shapes ${KS[$ID % ${#KS[@]}]} \
    --strides 1x1 \
    --input_shape ${INS[$ID % ${#INS[@]}]} \
    --n_kernels ${N_KS[$ID % ${#N_KS[@]}]} \
    --conv_act_fn relu \
    --h_units ${HS[$ID % ${#HS[@]}]} \
    --fc_act_fn relu \
    --out_act_fn sigmoid \
    --dropout ${DROPS[$ID % ${#DROPS[@]}]}




