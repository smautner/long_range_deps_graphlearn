#!/bin/bash
git commit -a -m 'autosync'
git push
ssh cluster 'cd /scratch/bi01/mautner/long_range_deps_graphlearn; git pull'
ssh bipc    'cd /scratch/1/mautner/long_range_deps_graphlearn; git pull'

