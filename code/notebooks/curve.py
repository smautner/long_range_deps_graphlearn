
# draw stuff
import matplotlib as mpl
mpl.use('Agg')

''' appearently this is unused :)
import eden
import matplotlib.pyplot as plt
from eden.util import configure_logging
import logging

from itertools import tee, chain, islice
import numpy as np
import random
from time import time
import datetime
from graphlearn.graphlearn import Sampler as GraphLearnSampler
from eden.util import fit,estimate
from eden.path import Vectorizer
import random
from eden.converter.graph.gspan import gspan_to_eden
from itertools import islice
import random
'''

'''
GET RNA DATA
'''
from eden.converter.fasta import fasta_to_sequence
import itertools
from Valium import sumsim

def rfam_uri(family_id):
    return 'http://rfam.xfam.org/family/%s/alignment?acc=%s&format=fastau&download=0'%(family_id,family_id)
def rfam_uri(family_id):
    return '%s.fa'%(family_id)
 

from eden.converter.fasta import fasta_to_sequence
def get_sequences_with_names(filename='RF00005.fa'):
    sequences = fasta_to_sequence("../toolsdata/"+filename)
    return sequences


def get_graphs(fname,size):
    graphs=[g for g in get_sequences_with_names(fname)]
    random.shuffle(graphs)
    return graphs[:size]




import random
import graphlearn.abstract_graphs.RNA as rna
from  graphlearn.feasibility import FeasibilityChecker as Checker
from graphlearn.estimator import Wrapper as estimatorwrapper
import graphlearn.utils.draw as draw
from graphlearn.graphlearn import Sampler as GLS
import itertools




from eden.converter.fasta import fasta_to_sequence
import matplotlib.pyplot as plt
# we always want a test and a train set to omou
def get_seq_tups(fname,size,sizeb):
    kram = fasta_to_sequence("../toolsdata/"+fname)
    graphs=[g for g in kram]
    random.shuffle(graphs)
    return graphs[:size],graphs[size:size+sizeb]

def plot(run_id, numgraphs, original_sample_repetitions, original_repetitions, sample_repetitions): # note that the var names are not real anymore.
    """
    """
    gc={'color':'g'}
    rc={'color':'r'}
    bc={'color':'b'}
    FONTSIZE=20
    ws = .3
    os = np.mean(original_sample_repetitions, axis=1)
    o = np.mean(original_repetitions, axis=1)
    s = np.mean(sample_repetitions, axis=1)
    plt.figure(figsize=(18,8))
    marksize=5


    # OKOK NEW STUFF TESTING
    fig, ax1 = plt.subplots()
    ax2=ax1.twinx()
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(15)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(15)
    numgraphs=np.array(numgraphs)

    #plt.grid()
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed')


    ax1.boxplot(original_sample_repetitions, positions=numgraphs - 0.4, widths=ws, capprops=gc, medianprops=gc, boxprops=gc, whiskerprops=gc, flierprops=None)
    ax1.boxplot(original_repetitions, positions=numgraphs, widths=ws, capprops=rc, medianprops=rc, boxprops=rc, whiskerprops=rc, flierprops=None)
    ax1.boxplot(sample_repetitions, positions=numgraphs + .4, widths=ws, capprops=bc, medianprops=bc, boxprops=bc, whiskerprops=bc, flierprops=None)
    ax1.plot(numgraphs, os, color='g', marker='o', markeredgewidth=1, markersize=marksize, markeredgecolor='g', markerfacecolor='w', label='original',linewidth=2)
    ax1.plot(numgraphs, o, color='r', marker='o', markeredgewidth=1, markersize=marksize, markeredgecolor='r', markerfacecolor='w', label='sample',linewidth=2)
    ax1.plot(numgraphs, s, color='b', marker='o', markeredgewidth=1, markersize=marksize, markeredgecolor='b', markerfacecolor='w', label='both',linewidth=2)


    global similarity_scores
    ax2.plot(numgraphs, similarity_scores, 'mo', markersize=marksize,label='similarity')
    #print 'similarity_scores = %s' % similarity_scores

    #plt.xlim(percentages[0]-.05,percentages[-1]+.05)
    print numgraphs
    plt.xlim(min(numgraphs)-2,max(numgraphs)+2)
    ax1.set_ylim(0.85,1.000)
    ax2.set_ylim(0.95,1.100)
    plt.xticks(numgraphs,numgraphs)
    '''
    ax = plt.subplot()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(20)

    plt.grid()
    plt.boxplot(original_sample_repetitions, positions=percentages, widths=ws, capprops=gc, medianprops=gc, boxprops=gc, whiskerprops=gc, flierprops=gc)
    plt.plot(percentages,os, color='g', marker='o', markeredgewidth=1, markersize=7, markeredgecolor='g', markerfacecolor='w', label='original')

    plt.boxplot(original_repetitions, positions=percentages, widths=ws, capprops=rc, medianprops=rc, boxprops=rc, whiskerprops=rc, flierprops=rc)
    plt.plot(percentages,o, color='r', marker='o', markeredgewidth=1, markersize=7, markeredgecolor='r', markerfacecolor='w', label='sample')

    plt.boxplot(sample_repetitions, positions=percentages, widths=ws, capprops=bc, medianprops=bc, boxprops=bc, whiskerprops=bc, flierprops=bc)
    plt.plot(percentages,s, color='b', marker='o', markeredgewidth=1, markersize=7, markeredgecolor='b', markerfacecolor='w', label='sample+orig')


    global similarity_scores
    plt.plot(percentages,similarity_scores, 'ro')
    print 'similarity_scores = %s' % similarity_scores

    plt.xlim(percentages[0]-.05,percentages[-1]+.05)
    plt.xlim(17,52)
    plt.ylim(0.7,1.100)
    '''

    #plt.title(run_id + '\n', fontsize=18)
    ax1.legend(loc='lower left',fontsize=14)
    ax2.legend(loc='lower right',fontsize=14)
    #plt.ylabel('ROC AUC',fontsize=18)
    ax1.set_ylabel('ROC AUC',fontsize=18)
    ax2.set_ylabel('similarity of instances',fontsize=18)
    plt.xlabel('Training set size per family',fontsize=18)
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
        ax1.plot(numgraphs.tolist()*9, tmpdata,
                 color=color,
                 marker='o',
                 markeredgewidth=1,
                 markersize=marksize,
                 markeredgecolor=color,
                 markerfacecolor='w',
                 linewidth=0)
        # plot the calculated line
        print 'means,ng',numgraphs, means
        tmpx = numgraphs.tolist()*9
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
    ax1.set_ylim(0.85,1.000)
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

#load("DATAS")
#plot("RF00162 vs RF00005 learning curve", [30,70], [[.30,.30],[.20,.20]] , [[.40,.40],[.30,.30]],[[.70,.35],[.25,.25]])
import random

import graphlearn.abstract_graphs.RNA as rna
from graphlearn.estimator import Wrapper as estimatorwrapper
from graphlearn.graphlearn import Sampler as GLS
from  graphlearn.feasibility import FeasibilityChecker as Checker
import graphlearn.utils.draw as draw
import itertools




def make_argsarray():
    args=[
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
    '''
    for improving_threshold in [ .3,.4,.5,.6]:
        for imp_lin_start in [.1,.2]:
            for max_size_diff in [10,15,20]:
                for acc_min_sim in [.50,.55,.60]:
                    if improving_threshold > imp_lin_start:
                        argz={}

                        argz['imp_thresh']=improving_threshold
                        argz['imp_lin_start']=imp_lin_start
                        argz['maxsizediff']=max_size_diff
                        argz['acc_min_sim']=acc_min_sim
                        args.append(argz)
                        '''
    return args
    

def fit_sample(graphs, random_state=random.random()):
    '''
    graphs -> more graphs
    arguments are generated above Oo
    '''
    global arguments
    NJOBS=1

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
size_test=20
dataset_a='RF00005.fa'
#dataset_a='RF01725.fa' 5 vs 162 was in the original paper 
dataset_b='RF00162.fa'
#sizes=[7,8,9,10,11,12,13,14,15]
sizes=range(20,55,5)

#sizes=[20,25]
repeats=9

# calc everything
def get_results():
    li = [ get_datapoint(size) for size in sizes ]
    # transpose , should work OO
    li =  [list(i) for i in zip(*li)]
    return li

# calc for one "size", go over repeats
def get_datapoint(size):
    ra=[]
    rb=[]
    rab=[]
    similarities=[]
    global similarity_scores
    for rep in range(repeats):
        a,b,ab,similarity = evaluate_point_no_deepcopy(size)
        ra.append(a)
        rab.append(ab)
        rb.append(b)
        similarities.append(similarity)
    similarity_scores.append( (sum(similarities)/float(len(similarities))))
    return ra,rb,rab



def evaluate_point(size):
    res=[]
    train_aa,train_a,test_a = get_trainthings(size,dataset_a)
    train_bb,train_b,test_b = get_trainthings(size,dataset_b)
    res.append(  test(deepcopy(train_a),deepcopy(train_b),deepcopy(test_a),deepcopy(test_b)) )
    eins=sumsim.get_similarity(deepcopy(train_aa),deepcopy(train_a))
    zwei=sumsim.get_similarity(deepcopy(train_bb),deepcopy(train_b))
    drei = (eins+zwei)/2.0
    res.append(  test(deepcopy(train_aa),deepcopy(train_bb),deepcopy(test_a),deepcopy(test_b)) )
    res.append(  test(deepcopy(train_a)+deepcopy(train_aa),deepcopy(train_b)+train_bb,deepcopy(test_a),deepcopy(test_b)) )
    res.append(drei)
    return res

def evaluate_point_no_deepcopy(size):
    res=[]
    train_aa,train_a,test_a = get_trainthings(size,dataset_a)
    train_bb,train_b,test_b = get_trainthings(size,dataset_b)
    res.append(  test(train_a,train_b,test_a,test_b) )
    eins=sumsim.get_similarity(train_aa,train_a)
    zwei=sumsim.get_similarity(train_bb,train_b)
    drei = (eins+zwei)/2.0
    res.append(  test(train_aa,train_bb,test_a,test_b) )
    res.append(  test(train_a+train_aa,train_b+train_bb,test_a,test_b) )
    res.append(drei)
    return res


# does the fit stuff
def get_trainthings(size,dataset):
    try:
        train,test = get_seq_tups(dataset,size,size_test)
        res=fit_sample(deepcopy(train))
        if len(res)<3:
            raise ValueError('wtf')
    except:
        print '.',
        return get_trainthings(size,dataset)
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
# MAIN
###########
if __name__ == "__main__":
    global similarity_scores
    similarity_scores=[]

    global arguments
    import sys
    arguments=[]
    argz = make_argsarray()
    #print argz
    print 'len argz',len(argz)
    print sys.argv

    # subtract one because sge is sub good
    job = int(sys.argv[1])-1 
    print 'jobid:',job
    arguments=argz[job]
    r=get_results()
    print 'sizes = %s' % sizes
    print 'result = %s' % r
    print 'similarity_scores = %s' % similarity_scores
    plot(str(job), sizes, *r)





