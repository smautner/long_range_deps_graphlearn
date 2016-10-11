#!/bin/bash
echo "##################################"
echo "#  CHECK PATHS IN THIS FILE       "
echo "##################################"
PYTHONPATH="/home/ikea/nips2016/code/deps/GraphLearn:$PYTHONPATH"
PYTHONPATH="/home/ikea/nips2016/code/deps/EDeN:$PYTHONPATH"
echo $PYTHONPATH
jupyter notebook  --no-browser

