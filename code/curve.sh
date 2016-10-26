#!/bin/bash
#$ -cwd
#$ -pe smp 2
#$ -l h_vmem=6G
#$ -M mautner@cs.uni-freiburg.de
#$ -o /home/mautner/JOBZ/curve_o/$JOB_NAME.$JOB_ID_o_$TASK_ID
#$ -e /home/mautner/JOBZ/curve_e/$JOB_NAME.$JOB_ID_e_$TASK_ID


#echo "##################################"
#echo "#  CHECK PATHS IN THIS FILE       "
#echo "##################################"
cd "notebooks"; python curve.py $SGE_TASK_ID
