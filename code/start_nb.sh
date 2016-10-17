#!/bin/bash
echo "##################################"
echo "#  CHECK PATHS IN THIS FILE       "
echo "##################################"

CODEPATH="$PWD/scratch/nips2016/code"
CODEPATH="$PWD"
PYTHONPATH="$CODEPATH/deps/GraphLearn:$PYTHONPATH"
PYTHONPATH="$CODEPATH/deps/EDeN:$PYTHONPATH"
PYTHONPATH="$CODEPATH/deps:$PYTHONPATH"
#MPLCONFIGDIR="/home/mautner/mylittlepony/matplotlib$SGE_TASK_ID/crap"

echo $PYTHONPATH
jupyter notebook  --no-browser

