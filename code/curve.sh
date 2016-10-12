#!/bin/bash
#$ -l h_vmem=6G
#$ -l cpu=4 
#echo "##################################"
#echo "#  CHECK PATHS IN THIS FILE       "
#echo "##################################"

echo "i hope the cpu option works :) "
cd "$CODEPATH/notebooks"; python curve.py $SGE_TASK_ID

