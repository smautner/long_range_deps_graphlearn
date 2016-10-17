#!/bin/bash
#$ -l h_vmem=6G
#$ -M mautner@cs.uni-freiburg.de
#echo "##################################"
#echo "#  CHECK PATHS IN THIS FILE       "
#echo "##################################"
#
CODEPATH="$PWD/scratch/nips2016/code"
CODEPATH="$PWD"
PYTHONPATH="$CODEPATH/deps/GraphLearn:$PYTHONPATH"
PYTHONPATH="$CODEPATH/deps/EDeN:$PYTHONPATH"
PYTHONPATH="$CODEPATH/deps:$PYTHONPATH"
#MPLCONFIGDIR="/home/mautner/mylittlepony/matplotlib$SGE_TASK_ID/crap"

# curve.py got its size=10 parameter fixed so remove that for serious stuff
# we need to do this becuase the cluster sucks with env vars :(
export PYTHONPATH
export CODEPATH
echo "i hope the cpu option works :) "
cd "$CODEPATH/notebooks"; python curve.py 1

