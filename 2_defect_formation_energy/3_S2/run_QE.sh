#!/bin/sh
#PBS -N qe_si
#PBS -V              
#PBS -q debug
#PBS -A qe
#PBS -l select=1:ncpus=68:mpiprocs=16:ompthreads=1
#PBS -l walltime=01:00:00

cd $PBS_O_WORKDIR

module purge
module load craype-mic-knl
module load intel/19.0.5 impi/19.0.5
module load qe/7.2

# 계산 실행 — INPUT_FILE만 바꿔가며 재사용
INPUT=s2.relax.in
OUTPUT=${INPUT%.in}.out
mpirun -np 16 pw.x -in $INPUT > $OUTPUT

