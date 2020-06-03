#!/bin/bash
#SBATCH -J dl_train
#SBATCH --export=USERAPPL,WRKDIR,LD_LIBRARY_PATH
#SBATCH --chdir=/wrk/users/eliel/projects/dl_course20/
#SBATCH -o /wrk/users/eliel/projects/dl_course20/jobs/res/%A_%a.txt
#SBATCH -e /wrk/users/eliel/projects/dl_course20/jobs/err/%A_%a.txt
#SBATCH -t 0-12:00:00
#SBATCH -c 10
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem=50G
#SBATCH --mail-type=END
#SBATCH --mail-user=eliel.soisalon-soininen@helsinki.fi

# --mem 10G

# interactive
# srun -t 10:00:00 --mem=10G -p gpu-short --gres=gpu:1 -c 10 --pty bash


module purge
module load Python/3.7.0-intel-2018b
module load CUDA/10.1.105


echo "training cnn for DL"

ID=SLURM_ARRAY_TASK_ID

EMBS=(enc=word2vec)
KS=("100X10 3x2" "150x10 2x2" "50x10 6x2")
PS=("1x9 1x5" "1x9 1x5" "1x9 1x5")
STS=("1x1" "1x1" "1x1")
N_KS=(100)
NC=(2)
MODS=(DocCNN)
INS=(300x100 300x100 768x100)
BS=(64)
OPTS=(adadelta)
HS=(100)
DROPS=(0.5)
# train model
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
    --pool_sizes ${PS[$ID % ${#PS[@]}]} \
    --strides ${STS[$ID % ${#STS[@]}]} \
    --input_shape ${INS[$ID % ${#INS[@]}]} \
    --n_kernels ${N_KS[$ID % ${#N_KS[@]}]} \
    --conv_act_fn relu \
    --h_units ${HS[$ID % ${#HS[@]}]} \
    --fc_act_fn relu \
    --out_act_fn sigmoid \
    --dropout ${DROPS[$ID % ${#DROPS[@]}]}




