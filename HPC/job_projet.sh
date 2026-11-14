#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1

# Setting up software environment
module load gcc openmpi scalasca miniconda3 openblas
conda activate hpc-python


# Application execution

for i in 1 2 3 4 5
do  
    python3 dr_solve.py
    python3 dr_solve_dd.py
done