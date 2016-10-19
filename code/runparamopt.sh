#!/bin/bash
source scratch/nips2016/code/setenv.sh 

#qsub -V -t 1-72 -pe smp 6  scratch/nips2016/code/curve.sh
qsub -V -t 1-100 scratch/nips2016/code/paramopt.sh
