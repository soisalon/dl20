#!/bin/bash
#SBATCH -J dl_train
#SBATCH --export=WRKDIR,LD_LIBRARY_PATH
#SBATCH --chdir=/wrk/users/eliel/projects/dl20/src
#SBATCH -o /wrk/users/eliel/projects/dl20/jobs/res/%J.txt
#SBATCH -e /wrk/users/eliel/projects/dl20/jobs/err/%J.txt
#SBATCH -t 0-10:00:00
#SBATCH -c 5
#SBATCH --mem=5G

#-p gpu
#--gres=gpu:1

# --mem-per-cpu=10g


# for interactive session:
# srun -t 0-4:00:00 -c 2 --mem=5G -p gpu-short --gres=gpu:1 --pty bash


module purge
module load Python/3.7.0-intel-2018b
# which version of cuda to install? 10.2
#module load CUDA/10.1.105

### How to use GPU - download CUDA module ?


echo "running data.sh"

# --array=0-5
#ID=SLURM_ARRAY_TASK_ID

srun $WRKDIR/ve37/bin/python3 -m data
