#!/bin/bash
#SBATCH --mail-user=fariaswa@ualberta.ca
#SBATCH --mail-type=ALL
#SBATCH --account=def-cepp
#SBATCH --array=1-60
#SBATCH --time=13:00:00
#SBATCH --mem=8192M
#SBATCH --output=slurm-%x.%j.out
#SBATCH --error=slurm-%x.%j.err
module load python/3.7
source env/bin/activate
python rnn_pipeline.py -f hyperparams/input.$SLURM_ARRAY_TASK_ID
