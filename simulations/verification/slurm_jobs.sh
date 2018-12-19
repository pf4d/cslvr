#!/bin/bash

##SBATCH --ntasks=36
#SBATCH --nodes=1
#SBATCH --job-name=poisson
#SBATCH -t 30
#SBATCH -o poisson.outlog
#SBATCH -e poisson.errlog
#SBATCH -p smp
#SBATCH --get-user-env

#module load centoslibs

#ldd $DOLFIN_DIR/lib/python2.7/site-packages/dolfin/cpp/_common.so;

for ((i=2; i<=16; i*=2)); do
	srun -n $i python ismip_hom_a.py $(($i * 10))
done

