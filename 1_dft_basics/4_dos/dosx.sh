#!/bin/sh 
#PBS -N qe_4_dosx 
#PBS -q debug 
#PBS -A qe 
#PBS -l select=1:ncpus=68:mpiprocs=16:ompthreads=1 
#PBS -l walltime=00:10:00 

cd $PBS_O_WORKDIR 
module purge 
module load craype-mic-knl intel/19.0.5 impi/19.0.5 qe/7.2 
mpirun -np 16 dos.x -in si.dos.in > si.dos.out

