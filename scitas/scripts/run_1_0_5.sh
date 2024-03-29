#!/bin/bash -l

# Run a job on 1 core + 1 GPU

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --account=transpor

module load gcc/8.4.0-cuda  cuda/11.2.2 cudnn/8.1.1.33-11.2-linux-x64 python/3.7.7
source ~/datgan_env/bin/activate
echo STARTING AT `date`
srun python train.py -n 5 -b 1 -ci 0
echo FINISHED at `date`