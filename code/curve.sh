#!/bin/bash
#$ -cwd
#$ -pe smp 2
#$ -l h_vmem=6G
#$ -M mautner@cs.uni-freiburg.de
#$ -o /home/mautner/JOBZ/$JOB_NAME.$JOB_ID_o_$TASK_ID.o
#$ -e /home/mautner/JOBZ/$JOB_NAME.$JOB_ID_e_$TASK_ID.e


#echo "##################################"
#echo "#  CHECK PATHS IN THIS FILE       "
#echo "##################################"
cd "$CODEPATH/notebooks"; python curve.py $TASK_ID
