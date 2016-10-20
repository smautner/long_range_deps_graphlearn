#!/bin/bash
#$ -cwd
#$ -pe smp 2
#$ -l h_vmem=4G
#$ -M mautner@cs.uni-freiburg.de
cd notebooks; python parametersearch.py

