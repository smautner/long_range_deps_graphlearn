#!/bin/bash
#$ -cwd
#$ -pe smp 2
#$ -l h_vmem=4G
cd notebooks; python parametersearch.py

