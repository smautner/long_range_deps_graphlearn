#!/bin/bash
#$ -cwd
#$ -pe smp 2
#$ -l h_vmem=6G
#$ -M mautner@cs.uni-freiburg.de
#$ -o /home/mautner/JOBZ_infcla/infernal_o/$JOB_NAME.$JOB_ID_o_$TASK_ID
#$ -e /home/mautner/JOBZ_infcla/infernal_e/$JOB_NAME.$JOB_ID_e_$TASK_ID


#echo "##################################"
#echo "#  CHECK PATHS IN THIS FILE       "
#echo "##################################"
cd "notebooks"; python -u infernal_classic.py $SGE_TASK_ID
