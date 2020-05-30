#!/bin/bash
#SBATCH -J dl_train
#SBATCH --export=USERAPPL,WRKDIR,LD_LIBRARY_PATH
#SBATCH --chdir=/wrk/users/eliel/projects/dl_course20/
#SBATCH -o /wrk/users/eliel/projects/dl_course20/jobs/res/%A_a.txt
#SBATCH -e /wrk/users/eliel/projects/dl_course20/jobs/err/%A_a.txt
#SBATCH -t 2-0
#SBATCH -c 2
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=eliel.soisalon-soininen@helsinki.fi

#SBATCH --array=0-5
#SBATCH --mem-per-cpu=10G
#SBATCH --ntasks=6

# --mem 10G

# interactive
# srun -t 10:00:00 --mem=10G -p gpu-short --gres=gpu:1 -c 10 --pty bash


module purge
module load Python/3.7.0-intel-2018b
module load CUDA/10.1.105


echo "training cnn for DL"

ID=SLURM_ARRAY_TASK_ID

EMBS=("elmo_2x1024_128_2048cnn_1xhighway dim=2" enc=bert-base-uncased)
KS=(256X2 768x2)
N_KS=(10)
NC=(1)
MODS=(BaseCNN)
INS=(256x100 768x100)
BS=(32 32 64 64 128 128)
OPTS=(adadelta)
HS=(100)
DROPS=(0.5)
# train model
srun -n 10 --exclusive $USERAPPL/ve37/bin/python3 dl20/src/train.py \
    --tr_ratio 0.2 \
    --dev_ratio 0.1 \
    --seed 100 \
    --cv_folds 1 \
    --emb_pars ${EMBS[$ID % ${#EMBS[@]}]} \
    --n_epochs 10 \
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



