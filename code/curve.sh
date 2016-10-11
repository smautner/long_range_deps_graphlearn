#!/bin/bash
#$ -l h_vmem=6G

#echo "##################################"
#echo "#  CHECK PATHS IN THIS FILE       "
#echo "##################################"

codepath="$PWD/scratch/nips2016/code"
PYTHONPATH="$codepath/deps/GraphLearn:$PYTHONPATH"
PYTHONPATH="$codepath/deps/EDeN:$PYTHONPATH"
#echo $PYTHONPATH
cd notebooks; python curve.py $SGE_TASK_ID

