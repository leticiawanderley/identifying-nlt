#!/bin/sh
#SBATCH --mail-user=fariaswa@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --account=def-cepp
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8192M
#SBATCH --output=slurm-%x.%j.out
#SBATCH --error=slurm-%x.%j.err
module load python/3.7
source env/bin/activate
python rnn_pipeline.py -lr 0.00001 -nh 32 -tc incorrect_ud_tags_padded -lo BCEwithLL -mb 64