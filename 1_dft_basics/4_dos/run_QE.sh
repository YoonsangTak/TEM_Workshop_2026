#!/bin/sh
#PBS -N qe_4_dos
#PBS -V              
#PBS -q gachon
#PBS -A qe
#PBS -l select=1:ncpus=16:mpiprocs=4:ompthreads=4
#PBS -l walltime=01:00:00

cd $PBS_O_WORKDIR

module purge
module load craype-mic-knl
module load intel/19.0.5 impi/19.0.5
module load qe/7.2

# 계산 실행 — INPUT_FILE만 바꿔가며 재사용
INPUT=si.nscf.in
OUTPUT=${INPUT%.in}.out
mpirun -np 16 pw.x -in $INPUT > $OUTPUT

