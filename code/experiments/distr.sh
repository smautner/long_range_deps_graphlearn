echo "{'fastafile': 'RF01725.fa', 'mininterfacecount': 2, 'burnin': 4, 'acc_min_sim': 0.24449402485485644, 'imp_lin_start': 0.19892265815047983, 'maxsizediff': 6, 'imp_thresh': 0.32120431812249317, 'mincipcount': 2, 'SCORE': -0.0, 'core_choice': False, 'n_samples': 10, 'n_steps': 25, 'quick_skip': True}" > test_distribution
cat test_distribution | python distribution.py  > tmp 
cat tmp | ./draw.py distribution
