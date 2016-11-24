


import subprocess
def shell_exec(cmd):
    #print "\n"+cmd+"\n"
    process = subprocess.Popen(cmd,stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, stderr = process.communicate()
    retcode = process.poll()
    return (retcode,stderr,output)



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

def fasta_to_list(fname):
    return [e for e in fasta_to_sequence(fname)]

def get_seq_tups(fname,size,sizeb):
    graphs=fasta_to_list('../toolsdata/'+fname)
    random.shuffle(graphs)
    return graphs[:size],graphs[size:size+sizeb]


def transpose(li):
    return [list(i) for i in zip(*li)]

##############################3
# the rest of this is dedicated to the infernal sampler
############################
def getstr(str):
    str=str.split("\n")
    second_line=str[1]
    return second_line.split(' ')[0]

def to_stockholm(fastaalifile, secondary_structure, outfile):
    seqs=fasta_to_list(fastaalifile)
    res="# STOCKHOLM 1.0\n\n#=GF SQ   %d\n\n" % len(seqs)
    for header,sequence in seqs:
        res+="%s       %s\n" % (header,sequence)
    res+="#=GC SS_cons        %s\n//\n" % secondary_structure
    with open(outfile,'w') as file:
        file.write(res)

def fit_sample_infernal(seques,dummy):
    """
    ok wir machen
     write fasta
     muscle,
     alifold,
     biopython,
     create_cm und cmemit oO
    """
    #print seques
    sequences = [b for a,b in seques]
    rna.write_fasta(sequences,"tmp.fa")
    shell_exec('muscle -in tmp.fa -out museld.fa')
    a,b,out = shell_exec('cat museld.fa | RNAalifold -f F --noPS')
    ss= getstr(out)
    to_stockholm('museld.fa',ss, 'sto.sto')
    shell_exec("cmbuild -F mod3l sto.sto")
    shell_exec("cmemit -N %d --exp 3.92  mod3l > out.fa" % (len(sequences)*2))
    return fasta_to_list('out.fa')





if __name__ == '__main__':
    seqs,trash=get_seq_tups("RF00005.fa",20,1)
    fit_sample_infernal(seqs)




