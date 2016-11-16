#!/bin/bash
#$ -cwd
#$ -pe smp 2
#$ -l h_vmem=4G
#$ -M mautner@cs.uni-freiburg.de
#$ -o /home/mautner/JOBZ/paramopt_o/$JOB_NAME.$JOB_ID_o_$TASK_ID
#$ -e /home/mautner/JOBZ/paramopt_e/$JOB_NAME.$JOB_ID_e_$TASK_ID


cd experiments; python -u optimizer.py

