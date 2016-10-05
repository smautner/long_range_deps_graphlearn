#!/bin/bash
PYTHONPATH="/scratch/1/mautner/nips2016/code/deps/GraphLearn:$PYTHONPATH"
PYTHONPATH="/scratch/1/mautner/nips2016/code/deps/EDeN:$PYTHONPATH"
echo $PYTHONPATH
jupyter nbconvert --to=html --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=None notebooks/curve.ipynb
