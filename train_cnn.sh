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
#SBATCH --mem=100G
#SBATCH --mail-type=END
#SBATCH --mail-user=eliel.soisalon-soininen@helsinki.fi

# --mem 10G


module purge
module load Python/3.7.0-intel-2018b
module load CUDA/10.1.105


echo "training cnn for DL"

ID=SLURM_ARRAY_TASK_ID

EMBS=(enc=word2vec)
KS=("100x10 10x2 6x1" "100x20 50x2 8x2" "200x20 6x2 10x2" "50x20 20x2 10x2 6x2" "31x21 20x4 10x2 2x1 4x1")
PS=("1x2 2x1 1x1" "2x2 4x4 1x5" "4x4 2x2 1x4" "4x4 2x2 2x2 1x3" "4x4 2x1 3x2 1x1 1x3")
STS=("10x10 1x1 1x1" "1x1 1x1 1x1" "1x1 1x1 1x1" "1x1 1x1 1x1 1x1" "1x1 1x1 1x1 1x1 1x1")
N_KS=("25 50 100" "25 50 100" "25 50 100" "25 50 100 200" "25 50 100 200 300")
NC=(3 3 3 4 5)
MODS=(DocCNN)
# INS=(300x100 300x100 768x100)
INS=(300x100)
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




