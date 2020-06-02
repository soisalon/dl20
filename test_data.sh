#!/bin/bash
#SBATCH -J dl_test_data
#SBATCH --export=USERAPPL,WRKDIR,LD_LIBRARY_PATH
#SBATCH --chdir=/wrk/users/eliel/projects/dl_course20/
#SBATCH -o /wrk/users/eliel/projects/dl_course20/jobs/res/%A_%a.txt
#SBATCH -e /wrk/users/eliel/projects/dl_course20/jobs/err/%A_%a.txt
#SBATCH -t 1-0
#SBATCH -c 10
#SBATCH --mail-type=END
#SBATCH --mail-user=eliel.soisalon-soininen@helsinki.fi
#SBATCH --mem=10G

# --gres=gpu:1
# -p gpu-short

#--mem-per-cpu=10G

# --ntasks=6



# interactive
# srun -t 10:00:00 --mem=10G -p gpu-short --gres=gpu:1 -c 10 --pty bash


module purge
module load Python/3.7.0-intel-2018b

# train model
# srun -n 4 --exclusive $USERAPPL/ve37/bin/python3 dl20/src/train.py \
srun $USERAPPL/ve37/bin/python3 dl20/get_data.py




