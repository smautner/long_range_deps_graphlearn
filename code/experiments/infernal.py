#!/usr/bin/python
import numpy as np
import graphlearn.abstract_graphs.RNA as rna

def evaluate(repeats, size, fitsample, RFAM,inputdict,debug):
    means = []
    stds = []
    for i in range(repeats):
        if debug: print 'start rep'
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


def evaluate_all(repeats,sizes,fitsample,RFAM,inputdict,debug):


    res= [evaluate(repeats,size,fitsample,RFAM,inputdict,debug) for size in sizes ]
    return utils.transpose(res)

    '''
    one=[]
    two=[]
    for size in sizes:
        a,b=evaluate(repeats,size,fitsample,RFAM,inputdict)
        one.append(a)
        two.append(b)
    return one,two
    '''
    # transpose and return
    #li=[one,two]
    #r= [list(i) for i in zip(*li)]
    #print r
    #print li
    #return li
# the plan is this: inf dictionaries generator > out
import sys
import getopt
import utils
if __name__ == "__main__":

    # PARSE THE ARGS:
    optlist, args = getopt.getopt(sys.argv[1:], '', ['rfam=', 'sizes=', 'repeats=', 'debug='])
    optlist=dict(optlist)

    defaults={'--rfam':'RF01725',
              '--sizes':"[10,20,50,100,200,400]",
              '--repeats': '7',
              '--debug':'False'}
    for k,v in defaults.items():
        if k not in optlist:
            optlist[k]=v
    for k,v in optlist.items():
        if k != '--rfam':
            optlist[k] = eval(v)

    ext = 'extgrammar' in args
    old = 'oldgrammar' in args
    inf = 'infernal' in args
    if old + ext + inf != 1:
        print "infernal needs olgrammar, newgrammar or infernal as argument"
        exit()
    if ext:
        sampler = utils.fit_sample
    elif old:
        sampler = utils.fit_sample_noabstr
    elif inf:
        sampler = utils.fit_sample_infernal

    # go over each line of input and start dropping picz
    for nth, line in enumerate(sys.stdin.readlines()):
        if line.strip():
            inputdict = eval(line)
            a,b=evaluate_all(optlist['--repeats'],
                             optlist['--sizes'],
                             sampler,
                             optlist['--rfam'],
                             inputdict,
                             optlist['--debug'])

            print optlist['--sizes'],',',a,',',b

