#!/bin/bash
#SBATCH -J dl_tr_final
#SBATCH --export=USERAPPL,WRKDIR,LD_LIBRARY_PATH
#SBATCH --chdir=/wrk/users/eliel/projects/dl_course20/
#SBATCH -o /wrk/users/eliel/projects/dl_course20/jobs/res/%A_%a.txt
#SBATCH -e /wrk/users/eliel/projects/dl_course20/jobs/err/%A_%a.txt
#SBATCH -t 0-10:00:00
#SBATCH -c 10
#SBATCH -p gpu-short
#SBATCH --gres=gpu:1
#SBATCH --mem=20G
#SBATCH --mail-type=END
#SBATCH --mail-user=eliel.soisalon-soininen@helsinki.fi

# --mem 10G


module purge
module load Python/3.7.0-intel-2018b
module load CUDA/10.1.105


echo "training final model and get test_preds"

# train model
srun $USERAPPL/ve37/bin/python3 dl20/src/train.py \
    --seed 100 \
    --use_seqs 1 \
    --final 1 \
    --emb_pars enc=word2vec \
    --n_epochs 20 \
    --batch_size 64 \
    --loss_fn bce \
    --optim adadelta \
    --opt_params default \
    --model_name DocCNN \
    --n_conv_layers 2 \
    --kernel_shapes 100x2 100x2 \
    --strides 1x1 1x1 \
    --pool_sizes 2x2 1x24 \
    --dilations 1x1 1x1 \
    --paddings 0x0 0x0 \
    --input_shape 300x100 \
    --n_kernels 100 100 \
    --conv_act_fn relu \
    --h_units 100 \
    --fc_act_fn relu \
    --out_act_fn sigmoid \
    --dropout 0.2




