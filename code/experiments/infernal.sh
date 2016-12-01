 echo "{'fastafile': 'RF01725.fa', 'mininterfacecount': 2, 'burnin': 4, 'acc_min_sim': 0.24449402485485644, 'imp_lin_start': 0.19892265815047983, 'maxsizediff': 6, 'imp_thresh': 0.32120431812249317, 'mincipcount': 2, 'SCORE': -0.0, 'core_choice': False, 'n_samples': 10, 'n_steps': 25, 'quick_skip': True}" > test_distribution
 echo "{'radius_list': [0, 1, 2], 'thickness_list': [2], 'mincip_count': 2, 'min_interfacecount': 3} " > test_infernal_oldgrammar
        
 cat test_infernal_oldgrammar | python infernal.py  oldgrammar > tmp2  
 cat test_distribution | python infernal.py extgrammar > tmp3  
 paste -d ',' tmp3 tmp2  > tmp4 
 cat tmp4 | ./draw.py infernal

