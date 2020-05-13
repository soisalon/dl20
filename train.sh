#!/usr/bin/env bash
#!/bin/bash
#SBATCH -J dl_train
#SBATCH --export=WRKDIR,LD_LIBRARY_PATH
#SBATCH --chdir=/wrk/users/eliel/projects/dl20/src
#SBATCH -o /wrk/users/eliel/projects/embeddia/embeddia-experiments/jobs/res/%J.txt
#SBATCH -e /wrk/users/eliel/projects/embeddia/embeddia-experiments/jobs/err/%J.txt
#SBATCH -t 1-0
#SBATCH -c 2
#SBATCH --mem=5G

#-p gpu
#--gres=gpu:1

# --mem-per-cpu=10g


# for interactive session:
# srun -t 0-4:00:00 -c 2 --mem=5G -p gpu-short --gres=gpu:1 --pty bash


module purge
module load Python/3.7.0-intel-2018b
# which version of cuda to install? 10.2
module load CUDA/10.1.105

### How to use GPU - download CUDA module ?


echo "running train_cnn.sh"

# --array=0-5
#ID=SLURM_ARRAY_TASK_ID


EMB_FILES=(\
finnish-elmo-sents.hdf5 \
swedish-elmo-sents.hdf5 \
english-elmo_2x1024_128_2048cnn_1xhighway-sents-train.hdf5 \
)

IND_FILES=(\
finnish-0.2-1-arts.txt \
swedish-0.2-1-arts.txt \
english-0.2-1-arts.txt \
)


echo "running train_cnn.sh"

# train model
srun $WRKDIR/ve37/bin/python3 -m main \
    --dev_ratio 0.2 \
    --test_ind_file english-0.2-1-arts.txt \
    --emb_file english-elmo_2x1024_128_2048cnn_1xhighway-tokens-train.hdf5 \
    --model pair-cnn \
    --elmo_dim 2 \
    --n_epochs 10 \
    --batch_size 64 \
    --loss_fn bce \
    --optim adadelta \
    --n_layers 1 \
    --kernel_shapes Hx4 \
    --strides 1x1 \
    --pool_sizes 1x2 \
    --input_width 30 \
    --n_filters 10 \
    --conv_act_fn relu \
    --merge_fn abs \
    --fc_units 64 \
    --fc_act_fn relu \
    --out_act_fn sigmoid \
    --dropout 0.5






