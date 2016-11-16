#!/usr/bin/python


import numpy as np

import graphlearn.abstract_graphs.RNA as rna

def evaluate(repeats, size, fitsample, RFAM,inputdict):
    means = []
    stds = []
    for i in range(repeats):

        sequences,void = utils.get_seq_tups(RFAM+'.fa',size,1)
        zz = fitsample(sequences,inputdict)
        # print zz[:3]
        zz=[b for a ,b in zz]
        result = rna.infernal_checker(zz, cmfile='../toolsdata/%s.cm' % RFAM,
                                      cmsearchbinarypath='../toolsdata/cmsearch')

        a = np.array(result)
        means.append(np.mean(a, axis=0))
        stds.append(np.std(a, axis=0))

    means.sort()
    stds.sort()
    #print (size, means, stds)
    return means[repeats / 2] * 100, stds[repeats / 2] * 100


def evaluate_all(repeats,sizes,fitsample,RFAM,inputdict):
    one=[]
    two=[]
    for size in sizes:
        a,b=evaluate(repeats,size,fitsample,RFAM,inputdict)
        one.append(a)
        two.append(b)
    # transpose and return
    li=[one,two]
    return [list(i) for i in zip(*li)]

# the plan is this: inf dictionaries generator > out
import sys
import getopt
import utils
if __name__ == "__main__":

    # PARSE THE ARGS:
    optlist, args = getopt.getopt(sys.argv[1:], '', ['rfam=', 'sizes=', 'repeats='])
    optlist=dict(optlist)
    if '--rfam' not in optlist:
        optlist['--rfam'] = "RF01725"
    optlist['--rfam']="\'"+optlist['--rfam']+"\'"
    if '--sizes' not in optlist:
        optlist['--sizes'] = "[10,20,50,100,200,400]"
    if '--repeats' not in optlist:
        optlist['--repeats'] = '7'
    for k, v in optlist.items():
        optlist[k] = eval(v)

    print optlist
    ext = 'extgrammar' in args
    old = 'oldgrammar' in args
    if old == ext:
        print "infernal needs olgrammar or newgrammar as argument"
        exit()
    if ext:
        sampler = utils.fit_sample
    else:
        sampler = utils.fit_sample_noabstr

    # go over each line of input and start dropping picz
    for nth, line in enumerate(sys.stdin.readlines()):
        if line.strip():
            inputdict = eval(line)
            a,b=evaluate_all(optlist['--repeats'],optlist['--sizes'],sampler,optlist['--rfam'],inputdict)
            print optlist['--sizes'],',',a,',',b

