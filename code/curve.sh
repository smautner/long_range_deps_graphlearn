#!/bin/bash
#$ -cwd
#$ -pe smp 2
#$ -l h_vmem=6G
#$ -M mautner@cs.uni-freiburg.de

#$ -o $HOME/JOBZ/$JOB_NAME.$JOB_ID/out/$TASK_ID.o
#$ -e $HOME/JOBZ/$JOB_NAME.$JOB_ID/err/$TASK_ID.e

#echo "##################################"
#echo "#  CHECK PATHS IN THIS FILE       "
#echo "##################################"
cd "$CODEPATH/notebooks"; python curve.py $TASK_ID
