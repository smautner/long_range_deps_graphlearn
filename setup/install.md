
```
conda env create -n longrange -f longrange.conda.yml
conda activate longrange
# pip install -r longrange.pip not needed apparently

mkdir  (pwd)/libs
conda env config vars set PYTHONPATH=(pwd)/libs

ln -s (realpath ..)/code/deps/GraphLearn/graphlearn/ libs/graphlearn
ln -s (realpath ..)/code/deps/EDeN/eden/ libs/eden
ln -s (realpath ..)/code/deps/Valium/ libs/Valium
```
