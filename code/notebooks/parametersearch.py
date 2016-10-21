
import eden
import Valium.sumsim as ss 
defaultarg={}

import graphlearn.abstract_graphs.RNA as rna
from graphlearn.estimator import Wrapper as estimatorwrapper

from graphlearn.graphlearn import Sampler as GLS
import random

NJOBS=1

def fit_sample(graphs, random_state=random.random(),arguments=defaultarg):
    '''
    graphs -> more graphs
    arguments are generated above Oo
    '''
    #print 'i am the sampler and i use this: '
    #print arguments
    graphs = list(graphs)
    estimator=estimatorwrapper( nu=.5, cv=2, n_jobs=NJOBS)
    sampler=rna.AbstractSampler(radius_list=[0,1],
                                thickness_list=[2], 
                                min_cip_count=arguments['mincipcount'], 
                                min_interface_count=arguments['mininterfacecount'], 
                                preprocessor=rna.PreProcessor(base_thickness_list=[1],
                                    ignore_inserts=True), 
                                postprocessor=rna.PostProcessor(),
                                estimator=estimator
                                #feasibility_checker=feasibility
                               )
    sampler.fit(graphs,grammar_n_jobs=NJOBS,grammar_batch_size=1)
    graphs = [ b for a ,b in graphs  ]
    graphs = sampler.sample(graphs,
                            n_samples=arguments['n_samples'],
                            batch_size=1,
                            n_steps=arguments['n_steps'],
                            n_jobs=NJOBS,
                            quick_skip_orig_cip=arguments['quick_skip'],
                            probabilistic_core_choice=arguments['core_choice'],
                            burnin=arguments['burnin'],
                            improving_threshold=arguments['imp_thresh'],
                            improving_linear_start=arguments['imp_lin_start'],
                            max_size_diff=arguments['maxsizediff'],
                            accept_min_similarity=arguments['acc_min_sim'],
                            select_cip_max_tries=30,
                            keep_duplicates=False,
                            include_seed=False,
                            backtrack=2,
                            monitor=False)
    result=[]
    for graphlist in graphs:
        result+=graphlist
    # note that this is a list [('',sequ),..]
    return result
    
    
import Valium.randset as rs
import curve
# das hier chillt in der curve rum
def get_data(size=10):
    return curve.get_seq_tups(fname='RF00005.fa',size=size,sizeb=50)

def run_and_score(argz):
    #print  "STARTED A RUN" # THIS IS THE NU DEBUG
    try:
        a,b= get_data(size=50)
        b=fit_sample(a,arguments=argz)
        a,b=ss.unpack(a,b)
        print "generated_seqs %d" % len(b)
        score = ss.score(a,b)
    except:
        print '.'
        return run_and_score(argz)
    return score


def meaning(argz,num=9):
    scores=[run_and_score(argz) for i in range(num)]
    scores.sort()
    #print scores
    return scores[num/2]
    
    
def zeloop():
    currenthigh=-2
    while True:
        argz=rs.get_random_params()
        res=meaning(argz,num=9)
        print res
        if currenthigh < res:
            currenthigh=res
            print '\n'+str(argz)

zeloop()   
    
