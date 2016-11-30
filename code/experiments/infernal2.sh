cat test_distribution | python distribution.py --sizes=[20,30] --repeats=3 > tmp ; 
cat tmp | ./draw.py distribution
cat test_distribution | python infernal.py  extgrammar > extgr ; 
echo "{}" | python infernal.py  infernal > inf ; 
paste -d ',' inf extgr  > tmp4 ; 
cat tmp4 | ./draw.py infernal
