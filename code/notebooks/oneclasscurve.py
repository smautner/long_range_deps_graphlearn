'''
GET RNA DATA
'''

from eden.converter.fasta import fasta_to_sequence
'''
def get_sequences_with_names(filename='RF00005.fa'):
    sequences = fasta_to_sequence("../toolsdata/"+filename)
    return sequences

def get_graphs(fname,size):
    graphs=[g for g in get_sequences_with_names(fname)]
    random.shuffle(graphs)
    return graphs[:size]
'''

# we always want a test and a train set to omou
def get_seq_tups(fname,size,sizeb):
    kram = fasta_to_sequence("../toolsdata/"+fname)
    graphs=[g for g in kram]
    random.shuffle(graphs)
    return graphs[:size],graphs[size:size+sizeb]





# draw stuff
import matplotlib as mpl
mpl.use('Agg')
from Valium import sumsim
import matplotlib.pyplot as plt

def plot(run_id, numgraphs, distribution, similarity): # note that the var names are not real anymore.
    """
    """
    rc={'color':'r'}
    bc={'color':'b'}
    ws = .3
    o = np.mean(distribution, axis=1)
    o = np.median(distribution, axis=1)
    s = np.mean(similarity, axis=1)
    s = np.median(similarity, axis=1)
    plt.figure(figsize=(18,8))
    marksize=5


    # OKOK NEW STUFF TESTING
    fig, ax1 = plt.subplots()
    ax2=ax1.twinx()
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(15)

    numgraphs=np.array(numgraphs)

    #plt.grid()
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed')

    ax1.boxplot(distribution, positions=numgraphs, widths=ws, capprops=bc, medianprops=bc, boxprops=bc, whiskerprops=bc, flierprops=None)
    ax2.boxplot(similarity, positions=numgraphs, widths=ws, capprops=rc, medianprops=rc, boxprops=rc, whiskerprops=rc, flierprops=None)
    ax1.plot(numgraphs, o, color='b', marker='o', markeredgewidth=1, markersize=marksize, markeredgecolor='b', markerfacecolor='w', label='KL divergence',linewidth=2)
    ax2.plot(numgraphs, s, color='r', marker='o', markeredgewidth=1, markersize=marksize, markeredgecolor='r', markerfacecolor='w', label='similarity',linewidth=2)


    #plt.xlim(percentages[0]-.05,percentages[-1]+.05)
    print numgraphs
    plt.xlim(min(numgraphs)-2,max(numgraphs)+2)
    ax1.set_ylim(0.0,1.000)
    ax2.set_ylim(0.6,1.100)
    plt.xticks(numgraphs,numgraphs)

    #plt.title(run_id + '\n', fontsize=18)
    ax1.legend(loc='lower left',fontsize=14)
    ax2.legend(loc='lower right',fontsize=14)
    #plt.ylabel('ROC AUC',fontsize=18)
    ax1.set_ylabel('divergence',fontsize=18)
    ax2.set_ylabel('similarity of instances',fontsize=18)
    ax2.set_xlabel('number of training sequences',fontsize=18)
    ax1.set_xlabel('number of training sequences',fontsize=18)
    plt.savefig('%s_plot_predictive_performance_of_samples.png' % run_id)


def learning_curve_function(x, a, b):
    return a * (1 - np.exp(-b * x))
from scipy.optimize import curve_fit
def plot2(run_id, numgraphs, original_sample, original, sample): # note that the var names are not real anymore.
    """
    drawing is buttugly... here we redraw it
        # mache die lines breiter und scikit-curviere sie .. wie in rnasynth/util
        https://github.com/fabriziocosta/RNAsynth/blob/master/evaluation/draw_utils.py
    """
    ws = .3
    plt.figure(figsize=(18,8))
    marksize=5


    # OKOK NEW STUFF TESTING
    fig, ax1 = plt.subplots()
    ax2=ax1.twinx()
    for ax in [ax1,ax2]:
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
            label.set_fontname('Arial')
            label.set_fontsize(15)
    numgraphs=np.array(numgraphs)

    #plt.grid()
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed')



    def draw_a_line(dataset, color='g',label='label'):

        means = np.mean(dataset, axis=1)
        std = np.std(dataset,axis=1)

        tmpdata = [e for li in dataset for e in li]
        # plot the dots
        numgrmultiplier  = len(tmpdata)/len(numgraphs.tolist()) # should be equal to the repeats...
        ax1.plot(numgraphs.tolist()*numgrmultiplier, tmpdata,
                 color=color,
                 marker='o',
                 markeredgewidth=1,
                 markersize=marksize,
                 markeredgecolor=color,
                 markerfacecolor='w',
                 linewidth=0)
        # plot the calculated line
        print 'means,ng',numgraphs, means
        tmpx = numgraphs.tolist()*numgrmultiplier
        a, b = curve_fit(learning_curve_function, tmpx, tmpdata)
        print 'a',a
        print 'b',b
        print '%'*80
        x_fit = np.linspace(numgraphs.min()-10, numgraphs.max()+10, 120)
        thing = learning_curve_function(x_fit, *a)
        ax1.plot(x_fit, thing, color+'-', label=label)
        #ax1.plot(numgraphs, means, 'r-', label=label)



    draw_a_line(sample,color='r',label='sample')
    draw_a_line(original,color='g',label='original')
    draw_a_line(original_sample,color='b',label='both')


    global similarity_scores
    ax2.plot(numgraphs, similarity_scores, 'mo', markersize=marksize,label='similarity')
    #print 'similarity_scores = %s' % similarity_scores

    #plt.xlim(percentages[0]-.05,percentages[-1]+.05)
    print numgraphs
    plt.xlim(min(numgraphs)-2,max(numgraphs)+2)
    ax1.set_ylim(0.70,1.000)
    ax2.set_ylim(0.95,1.100)
    plt.xticks(numgraphs,numgraphs)

    #plt.xlim(percentages[0]-.05,percentages[-1]+.05)
    #plt.xlim(17,52)
    #plt.ylim(0.7,1.100)

    #plt.title(run_id + '\n', fontsize=18)
    ax1.legend(loc='lower left',fontsize=14)
    ax2.legend(loc='lower right',fontsize=14)
    #plt.ylabel('ROC AUC',fontsize=18)
    ax1.set_ylabel('ROC AUC',fontsize=18)
    ax2.set_ylabel('similarity of instances',fontsize=18)
    plt.xlabel('Training set size per family',fontsize=18)
    plt.savefig('%s_plot_predictive_performance_of_samples.png' % run_id)

import random
import graphlearn.abstract_graphs.RNA as rna
from graphlearn.estimator import Wrapper as estimatorwrapper

def make_argsarray():
    args=[
        {'mininterfacecount': 2, 'burnin': 4, 'acc_min_sim': 0.24449402485485644, 'imp_lin_start': 0.19892265815047983, 'maxsizediff': 6, 'imp_thresh': 0.32120431812249317, 'mincipcount': 2, 'core_choice': False, 'n_samples': 10, 'n_steps': 25, 'quick_skip': True, 'SCORE':-0.000},
        {'mininterfacecount': 2, 'burnin': 13, 'acc_min_sim': 0.24449402485485644, 'imp_lin_start': 0.19892265815047983, 'maxsizediff': 17, 'imp_thresh': 0.32120431812249317, 'mincipcount': 2, 'core_choice': False, 'n_samples': 6, 'n_steps': 70, 'quick_skip': True, 'SCORE':-0.703362611732},
        {'mininterfacecount': 1, 'burnin': 11, 'acc_min_sim': 0.35723373060996666, 'imp_lin_start': 0.11639352115717616, 'maxsizediff': 12, 'imp_thresh': 0.34966775094400682, 'mincipcount': 2, 'core_choice': True, 'n_samples': 6, 'n_steps': 25, 'quick_skip': True, 'SCORE':-0.699739011906},
        {'mininterfacecount': 2, 'burnin': 2, 'acc_min_sim': 0.22989399280978964, 'imp_lin_start': 0.0077498579055246264, 'maxsizediff': 12, 'imp_thresh': 0.97485773117432351, 'mincipcount': 2, 'core_choice': True, 'n_samples': 3, 'n_steps': 86, 'quick_skip': True, 'SCORE':-0.698742678067},
        {'mininterfacecount': 2, 'burnin': 10, 'acc_min_sim': 0.33286327128488141, 'imp_lin_start': 0.24708994438513876, 'maxsizediff': 6, 'imp_thresh': 0.79082744383037717, 'mincipcount': 2, 'core_choice': True, 'n_samples': 3, 'n_steps': 97, 'quick_skip': False, 'SCORE':-0.696588858544},
        {'mininterfacecount': 1, 'burnin': 3, 'acc_min_sim': 0.57797674073374372, 'imp_lin_start': 0.086316471690329077, 'maxsizediff': 14, 'imp_thresh': 0.54820600763755556, 'mincipcount': 2, 'core_choice': False, 'n_samples': 3, 'n_steps': 33, 'quick_skip': False, 'SCORE':-0.696254759283},
        {'mininterfacecount': 1, 'burnin': 8, 'acc_min_sim': 0.33507959375875951, 'imp_lin_start': 0.3316248479960533, 'maxsizediff': 10, 'imp_thresh': 0.65922778063175635, 'mincipcount': 2, 'core_choice': False, 'n_samples': 7, 'n_steps': 99, 'quick_skip': False, 'SCORE':-0.695379434783},
        {'mininterfacecount': 2, 'burnin': 1, 'acc_min_sim': 0.73917049113151512, 'imp_lin_start': 0.09306413705722727, 'maxsizediff': 14, 'imp_thresh': 0.62998549481543387, 'mincipcount': 1, 'core_choice': False, 'n_samples': 2, 'n_steps': 73, 'quick_skip': True, 'SCORE':-0.693406445222},
        {'mininterfacecount': 2, 'burnin': 5, 'acc_min_sim': 0.4486388144723355, 'imp_lin_start': 0.09374179056766796, 'maxsizediff': 19, 'imp_thresh': 0.24993359270552518, 'mincipcount': 2, 'core_choice': True, 'n_samples': 7, 'n_steps': 99, 'quick_skip': False, 'SCORE':-0.690620968873},
        {'mininterfacecount': 2, 'burnin': 11, 'acc_min_sim': 0.59318328541230492, 'imp_lin_start': 0.1842925803111628, 'maxsizediff': 18, 'imp_thresh': 0.79905439891716812, 'mincipcount': 2, 'core_choice': True, 'n_samples': 6, 'n_steps': 49, 'quick_skip': False, 'SCORE':-0.68998713873},
        {'mininterfacecount': 1, 'burnin': 8, 'acc_min_sim': 0.62734080199879139, 'imp_lin_start': 0.10469662908481758, 'maxsizediff': 7, 'imp_thresh': 0.11177296372179102, 'mincipcount': 2, 'core_choice': False, 'n_samples': 5, 'n_steps': 91, 'quick_skip': True, 'SCORE':-0.688879734274}]

    #return args


    fastas=[['RF01051.fa','RF01998.fa'],
            ['RF00001.fa','RF00162.fa'],
            ['RF00020.fa','RF01999.fa'],
            ['RF01999.fa','RF02344.fa'],
            ['RF00020.fa','RF02344.fa'],
            ['RF01725.fa','RF00167.fa'],
            ['RF01750.fa','RF00167.fa'],
            ['RF01725.fa','RF01750.fa']]

    fastadi={}
    for a,b in fastas:
        fastadi[a]=0
        fastadi[b]=0

    realres=[]
    uniques=fastadi.keys()
    uniques.sort()
    for key in uniques:
        for d in args:
            z=d.copy()
            z['fastafile']=key
            realres.append(z)

    return realres

def fit_sample(graphs, random_state=random.random()):
    '''
    graphs -> more graphs
    arguments are generated above Oo
    '''
    global arguments


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






from copy import deepcopy
import numpy as np
#  ok erstmal ueber alle x values, ne


# calc everything
def get_results(repeats=7,sizes=[],argparam=-2,njobs=1):
    print "repeats %d ; sizes = %s ; argparam =%d ; fastafile= %s" % (repeats,sizes,argparam, dataset_a )
    global NJOBS
    NJOBS=njobs
    global arguments
    arguments=argz[argparam]
    li = [ get_datapoint(size,repeats) for size in sizes ]
    # transpose
    # get datapoints is giving back 3 points per size.  the transposition seperates the data again
    li =  [list(i) for i in zip(*li)]
    print li
    return li


# calc for one "size", go over repeats
def get_datapoint(size,repeats):
    resa=[]
    resb=[]
    for rep in range(repeats):
        a,b = evaluate_point(size)
        resa.append(a)
        resb.append(b)
    return resa,resb



def evaluate_point(size):
    new , train, test_a = get_trainthings(size,dataset_a)
    return sumsim.get_dist_and_sim_crossval(new,train,kfold=3)

# does the fit stuff
def get_trainthings_debug(size,dataset,depth=0):
        train,test = get_seq_tups(dataset,size,size_test)
        res=fit_sample(deepcopy(train))
        if len(res)<10:
            print res
            raise ValueError('wtf')
        else:
            print 'trained with success'
        return (res,train,test)

# does the fit stuff
def get_trainthings(size,dataset,depth=0):
    if depth==6:
        exit()
    try:
        train,test = get_seq_tups(dataset,size,size_test)
        res=fit_sample(deepcopy(train))
        if len(res)<10:
            raise ValueError('wtf')
    except:
        print '.',
        return get_trainthings(size,dataset,depth+1)
    print 'k',
    return (res,train,test)


#############
# test()  will evaluate 2 sets of graphs
##########
from sklearn.linear_model import SGDClassifier
from eden.path import Vectorizer
def train_esti(neg,pos):
    v=Vectorizer()
    matrix=v.transform(neg+pos)
    res=SGDClassifier(shuffle=True)
    res.fit(matrix, np.asarray(  [-1]*len(neg)+[1]*len(pos)  ) )
    return res

def eva(esti,ne,po):
    v=Vectorizer()
    matrix=v.transform(ne)
    correct= sum(  [1 for res in esti.predict(matrix) if res == -1] )
    matrix2=v.transform(po)
    correct+= sum(  [1 for res in esti.predict(matrix2) if res == 1] )
    return correct

def test(a,b,ta,tb):
    est=train_esti(a,b)
    correct=eva(est,ta,tb)
    return correct/float(size_test*2) # fraction correct



#############
# CONSTANTS AND MAIN
###########
size_test=1
#dataset_a='RF00005.fa'
#dataset_b='RF00162.fa' RF01051 RF01998
dataset_b='RF01051.fa'
dataset_a='RF01725.fa'

#sizes=[7,8,9,10,11,12,13,14,15]
sizes=range(20,55,5)
sizes=[25,50,75,100]
repeats=7
NJOBS=2

from eden.util import configure_logging
import logging
#configure_logging(logging.getLogger(),verbosity=2)


argz = make_argsarray()

import sys
if __name__ == "__main__":

    debug= False
    if 'debug' in sys.argv[1:]:
        debug=True
    if debug:
        sizes = [20,30]
        repeats = 3
        NJOBS=4

    # set up task
    if debug: print 'choosing from x options:',len(argz)
    global arguments
    arguments=[]
    if debug: print 'argv:',sys.argv
    # subtract one because sge is sub good
    job = int(sys.argv[1])-1
    print 'jobid:',job
    dataset_a=argz[job]['fastafile']

    # look at res
    r=get_results(repeats=repeats,sizes=sizes,argparam=job,njobs=NJOBS)
    print 'sizes = %s' % sizes
    print 'result = %s' % r
    print 'similarity_scores = %s' % similarity_scores
    plot(str(job), sizes, *r)
    #plot2('p2_'+str(job), sizes, *r)



## NOTE TO SELF  CHECK NJOBS AND THE 524 where i overwrite the fastafilez. dataset_a= etc
'''   to make things parallel we should use this in evaluate point (e) .. also there is a mark on print
# yes, this already existed in the old eden

pool = mp.Pool()
mpres = [eden.apply_async(pool, mass_annotate_mp, args=(graphs, vectorizer, score_attribute, estimator)) for
         graphs in eden.grouper(inputs, 50)]
result = []
for res in mpres:
    result += res.get()
pool.close()
'''
