#!/bin/bash

# sbatch run_calib.sh /home/crrvcs/OpenKE/ transe 0 {norm, sigmoid} valid

#SBATCH -t 48:0:0

#SBATCH -A StaMp -p tier3 -n 1 -c 8

# Job memory requirements in MB
#SBATCH --mem=20000

#SBATCH --output=./LogsTrain/Calib_%A_%a.out
#SBATCH --error=./LogsTrain/Calib_%A_%a.err

folder=$1
modelName=$2
dataset=$3
norm=$4
type=$5

# Loop and submit all the jobs
echo " * Submitting job array..."

echo "SLURM_JOBID: " $SLURM_JOBID
echo "SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID
echo "SLURM_ARRAY_JOB_ID: " $SLURM_ARRAY_JOB_ID

/home/crrvcs/ActivePython-3.7/bin/python3 -Wignore -u ./Code/calib.py ${folder} ${modelName} ${dataset} ${norm} ${type}

echo " Done with job array"