#!/bin/bash
#source scratch/nips2016/code/setenv.sh 

mkdir -p /home/mautner/JOBZ/paramopt_o
mkdir -p /home/mautner/JOBZ/paramopt_e

source ./setenv.sh

#qsub -V -t 1-72 -pe smp 6  scratch/nips2016/code/curve.sh
qsub -V -t 1-100 parameter_optimisation.sh
