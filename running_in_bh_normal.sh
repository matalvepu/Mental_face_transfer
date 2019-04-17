#!/bin/sh


#SBATCH -p gpu --gres=gpu:1
#SBATCH --mem=20gb
#SBATCH -c 4
#SBATCH -a 41-50
#SBATCH -t 2-00:00:00  
#SBATCH -J ets_echowdh2
#SBATCH -o /scratch/mhasan8/output/ets_normal_output%j
#SBATCH -e /scratch/mhasan8/output/ets_normal_error%j
#SBATCH --mail-type=all    

module load anaconda3/5.3.0b
module load git
python running_different_configs.py --dataset=mental_face