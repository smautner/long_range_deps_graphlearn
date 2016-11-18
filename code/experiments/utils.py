






###############
## get samplers or data or whatever
###############



import graphlearn.abstract_graphs.RNAnoAbs as rnana
import graphlearn.abstract_graphs.RNA as rna
from graphlearn.estimator import Wrapper as estimatorwrapper
import random
def fit_sample_noabstr(sequences, arguments, random_state=random.random()):
    '''
    graphs -> more graphs
    graphs are pretty mich (NAME,SEQUENCE),()...
    '''
    # fit a sampler
    sequences = list(sequences)
    estimator = estimatorwrapper(nu=.5, cv=2, n_jobs=1)  # with .5 it also works for the fewer ones..
    sampler = rna.AbstractSampler(radius_list=arguments['radius_list'],  # [0, 1, 2],  # war 0,1
                                  thickness_list=arguments['thickness_list'],  # [1],  # war 2
                                  min_cip_count=arguments['mincip_count'],
                                  min_interface_count=arguments['min_interfacecount'],
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
    return result
    #return [r[1] for r in result]


def fit_sample(sequences, arguments,NJOBS=1, random_state=random.random()):
    '''
    graphs -> more graphs
    arguments are generated above Oo
    '''
    sequences = list(sequences)
    estimator = estimatorwrapper(nu=.5, cv=2, n_jobs=NJOBS)
    sampler = rna.AbstractSampler(radius_list=[0, 1],
                                  thickness_list=[2],
                                  min_cip_count=arguments['mincipcount'],
                                  min_interface_count=arguments['mininterfacecount'],
                                  preprocessor=rna.PreProcessor(base_thickness_list=[1],
                                                                ignore_inserts=True),
                                  postprocessor=rna.PostProcessor(),
                                  estimator=estimator
                                  # feasibility_checker=feasibility
                                  )
    sampler.fit(sequences, grammar_n_jobs=NJOBS, grammar_batch_size=1)
    sequences = [b for a, b in sequences]
    sequences = sampler.sample(sequences,
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
    result = []
    for graphlist in sequences:
        result += graphlist
    # note that this is a list [('',sequ),..]
    return result



# GET SEQUENCES, a sequence is actually a tupple (header,sequence)
from eden.converter.fasta import fasta_to_sequence
def get_seq_tups(fname,size,sizeb):
    kram = fasta_to_sequence("../toolsdata/"+fname)
    graphs=[g for g in kram]
    random.shuffle(graphs)
    return graphs[:size],graphs[size:size+sizeb]


def transpose(li):
    return [list(i) for i in zip(*li)]





