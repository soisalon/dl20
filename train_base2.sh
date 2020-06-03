#!/bin/bash
#SBATCH -J dl_train
#SBATCH --export=USERAPPL,WRKDIR,LD_LIBRARY_PATH
#SBATCH --chdir=/wrk/users/eliel/projects/dl_course20/
#SBATCH -o /wrk/users/eliel/projects/dl_course20/jobs/res/%A_%a.txt
#SBATCH -e /wrk/users/eliel/projects/dl_course20/jobs/err/%A_%a.txt
#SBATCH -t 0-16:00:00
#SBATCH -c 10
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END
#SBATCH --mail-user=eliel.soisalon-soininen@helsinki.fi
#SBATCH --mem=100G

#--mem-per-cpu=10G

# --ntasks=6



# interactive
# srun -t 10:00:00 --mem=10G -p gpu-short --gres=gpu:1 -c 10 --pty bash
# srun -t 10:00:00 --mem=10G -c 10 --pty bash

module purge
module load Python/3.7.0-intel-2018b
module load CUDA/10.1.105


echo "training cnn for DL"

ID=SLURM_ARRAY_TASK_ID
# ID=2

EMBS=("enc=elmo_2x1024_128_2048cnn_1xhighway dim=2" enc=bert-base-uncased enc=word2vec enc=glove)
KS=(256x2 768x2 300x2 300x2)
N_KS=(100)
NC=(1)
MODS=(BaseCNN)
INS=(256x1000 768x100 300x100 300x100)
BS=(32)
OPTS=(adadelta)
HS=(100)
DROPS=(0.5)
# train model
# srun -n 4 --exclusive $USERAPPL/ve37/bin/python3 dl20/src/train.py \
srun $USERAPPL/ve37/bin/python3 dl20/src/train.py \
    --dev_ratio 0.1 \
    --seed 100 \
    --use_seqs False \
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




