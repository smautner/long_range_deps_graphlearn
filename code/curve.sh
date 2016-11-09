#!/bin/bash
#$ -cwd
#$ -pe smp 3
#$ -l h_vmem=6G
#$ -M mautner@cs.uni-freiburg.de
#$ -o /home/mautner/JOBZ_c4/curve_o/$JOB_NAME.$JOB_ID_o_$TASK_ID
#$ -e /home/mautner/JOBZ_c4/curve_e/$JOB_NAME.$JOB_ID_e_$TASK_ID


#echo "##################################"
#echo "#  CHECK PATHS IN THIS FILE       "
#echo "##################################"
cd "notebooks"; python -u oneclasscurve.py $SGE_TASK_ID
