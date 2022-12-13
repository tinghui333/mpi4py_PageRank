#!/bin/bash
#
#SBATCH --job-name=tinghwu_job
#SBATCH --output=res.txt
#SBATCH --ntasks-per-node=4
#SBATCH --nodes=2
#SBATCH --time=10:00
#SBATCH -p short-40core

module load shared
module load anaconda/2
module load mvapich2/gcc/64/2.2rc1

start=$SECONDS

mpirun -np 4 python /gpfs/home/tinghwu/project3/project3_tinghui.py > output.log 
# mpirun -np 4 python /gpfs/projects/AMS598/class2022/tinghwu/project3/project3_tinghui.py > output.log 

echo  $((SECONDS - start))
