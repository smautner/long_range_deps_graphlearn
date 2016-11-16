'''
GET RNA DATA
'''
from eden.converter.fasta import fasta_to_sequence
import itertools
from eden.util import random_bipartition_iter,selection_iterator
import random

def rfam_uri(family_id):
    return 'http://rfam.xfam.org/family/%s/alignment?acc=%s&format=fastau&download=0'%(family_id,family_id)
def rfam_uri(family_id):
    return '%s.fa'%(family_id)


RFAM="RF01725"
#cutoff 162 (44.0)
#cutoff 1725 (38.0)
#cutoff rest (29)


def get_sequences_with_names(size=9999, rand=True):
    it = fasta_to_sequence("../toolsdata/%s.fa" % RFAM)
    it = list(it)
    if rand:
        #sequences , boring = random_bipartition_iter(it,.9,random_state=random.random())
        r=range(len(it))
        random.shuffle(r)
        return selection_iterator(it,r[:size])
    else:
        sequences = itertools.islice(it, size)
    return sequences


'''
SET UP THE SAMPLERS/// FITTING AND SAMPLING
'''
import random
from  graphlearn.feasibility import FeasibilityChecker as Checker
from graphlearn.estimator import Wrapper as estimatorwrapper
import graphlearn.utils.draw as draw
from graphlearn.graphlearn import Sampler as GLS
import itertools
import graphlearn.abstract_graphs.RNAnoAbs as rnana
import graphlearn.abstract_graphs.RNA as rna
# ok for the evaluation we need to
def fit_sample_noabstr(sequences,argz, random_state=random.random()):
    '''
    graphs -> more graphs
    graphs are pretty mich (NAME,SEQUENCE),()...
    '''

    # fit a sampler
    sequences = list(sequences)
    estimator = estimatorwrapper(nu=.5, cv=2, n_jobs=1)  # with .5 it also works for the fewer ones..
    sampler = rna.AbstractSampler(radius_list=argz['radius_list'],#[0, 1, 2],  # war 0,1
                                  thickness_list=argz['thickness_list'],#[1],  # war 2
                                  min_cip_count=argz['mincip_count'],
                                  min_interface_count=argz['min_interfacecount'],
                                  preprocessor=rnana.PreProcessor(base_thickness_list=[1], ignore_inserts=True),
                                  postprocessor=rna.PostProcessor(),
                                  estimator=estimator
                                  # feasibility_checker=feasibility
                                  )
    sampler.fit(sequences, grammar_n_jobs=1, grammar_batch_size=1)

    # logger.info('graph grammar stats:')
    dataset_size, interface_counts, core_counts, cip_counts = sampler.grammar().size()
    # logger.info('#instances:%d   #interfaces: %d   #cores: %d   #core-interface-pairs: %d' % (dataset_size, interface_counts, core_counts, cip_counts))






    sequences = [b for a, b in sequences]
    sequences = sampler.sample(sequences,
                              n_samples=5,
                              batch_size=1,
                              n_steps=55,
                              n_jobs=1,
                              quick_skip_orig_cip=True,
                              probabilistic_core_choice=False,
                              burnin=6,
                              improving_threshold=0.5,
                              improving_linear_start=0.15,
                              max_size_diff=6,
                              accept_min_similarity=0.55,
                              select_cip_max_tries=30,
                              keep_duplicates=False,
                              include_seed=False,
                              backtrack=2,
                              monitor=False)

    result = []
    for li in sequences:
       result += li
    return [r[1] for r in result]


'''
evaluation for one data-point
'''
import numpy as np


def evaluate(repeats, size, fitsample):
    print 'eval:',
    means = []
    stds = []
    for i in range(repeats):
        sequences = get_sequences_with_names(size=size, rand=10)
        zz = fitsample(sequences)
        # print zz[:3]
        # z=[b for a ,b in zz]
        result = rna.infernal_checker(zz, cmfile='../toolsdata/%s.cm' % RFAM,
                                      cmsearchbinarypath='../toolsdata/cmsearch')

        a = np.array(result)
        means.append(np.mean(a, axis=0))
        stds.append(np.std(a, axis=0))

    means.sort()
    stds.sort()
    print (size, means, stds)
    return [means[repeats / 2] * 100, stds[repeats / 2] * 100]




def getargsarray():
    args=[]
    for rl in [ [1,2],[2,3] ,[0,1,2]]:
        for tl in [[1,2],[1],[2]]:
            for mc in [2,3]:
                for mi in [2,3]:
                    argz={}
                    argz['radius_list'] = rl
                    argz['thickness_list'] =tl
                    argz['mincip_count'] = mc
                    argz['min_interfacecount'] =mi
                    args.append(argz)
    return args


args=getargsarray()


sizes=[10,20,50,100,200,400]
repeats=5


#sizes=[10,20]
#repeats=2


means2 = []
stds2 = []


if __name__ == "__main__":
    import sys
    job = int(sys.argv[1]) - 1
    for size in sizes:
        m, s = evaluate(repeats, size, lambda x: fit_sample_noabstr(x,args[job]) )
        means2.append(m)
        stds2.append(s)


    print "job = %d" % job
    print "means2 = %s" % means2
    print "stds2 = %s" % stds2
    print 'sum = %s' % sum(means2)