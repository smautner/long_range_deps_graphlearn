#!/bin/bash
git commit -a -m 'autosync'
git push
ssh cluster 'cd /scratch/bi01/mautner/nips2016; git pull'
