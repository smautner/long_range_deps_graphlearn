#!/bin/bash
#$ -cwd
#$ -pe smp 2
#$ -l h_vmem=6G
#$ -M mautner@cs.uni-freiburg.de

#$ -o /home/mautner/JOBZ/$JOB_NAME.$JOB_ID/out/$TASK_ID.o
#$ -e /home/mautner/JOBZ/$JOB_NAME.$JOB_ID/err/$TASK_ID.e

mkdir $HOME/JOBZ/$JOB_NAME.$JOB_ID/out
mkdir $HOME/JOBZ/$JOB_NAME.$JOB_ID/err

#echo "##################################"
#echo "#  CHECK PATHS IN THIS FILE       "
#echo "##################################"
cd "$CODEPATH/notebooks"; python curve.py $TASK_ID
