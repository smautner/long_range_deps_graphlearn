from copy import deepcopy
from Valium import sumsim
#plan says this:
#dictribution adict  > res1
#res1 > draw distribution
# calc everything


def get_results(args):
    li = [ get_datapoint(size,args) for size in args['--sizes'] ]
    # transpose
    # get datapoints is giving back 3 points per size.  the transposition seperates the data again
    li =  [list(i) for i in zip(*li)]
    return li


import multiprocessing as mp

def runpool(size,args):
    pool = mp.Pool()
    rep = int(args['--repeats'])
    results = pool.map(evaluate_point, [[size,args]]*rep )
    pool.close()
    pool.join()
    return results

# calc for one "size", go over repeats
def get_datapoint(size,args):
    # res = runpool(size,args)
    res=[evaluate_point((size,args)) for i in range(args['--repeats']) ]
    a,b=utils.transpose(res)
    return a,b

def evaluate_point(size_args):
    size,args = size_args
    new , train  = get_trainthings(size,args)
    res = sumsim.get_dist_and_sim_crossval(new,train,kfold=3)
    # print 'evaluate_point:', res
    return res


# does the fit stuff
def get_trainthings(size,args,depth=0):
    if depth==6:
        exit()

    # # 2024 block
    # train,test = utils.get_seq_tups(args['--fasta'],size,1)
    # res=utils.fit_sample(deepcopy(train),args,args['--njobs'])
    # return res,train

    try:
        train,test = utils.get_seq_tups(args['--fasta'],size,1)
        res=utils.fit_sample(deepcopy(train),args,args['--njobs'])
        if len(res)<10:
            raise ValueError('wtf')

    except Exception as excpt:
        if args['--debug']:
            print excpt
            print '.',
        return get_trainthings(size,args,depth+1)
    return (res,train)


import sys
import getopt
import utils
if __name__ == "__main__":

    # PARSE THE ARGS:
    optlist, args = getopt.getopt(sys.argv[1:], '', ['fasta=', 'sizes=', 'repeats=','njobs=','debug='])
    optlist=dict(optlist)
    # defaults={'--fasta':'RF01725.fa','--sizes':"[10,20,50,100,200,400]",
    # defaults={'--fasta':'RF01725.fa','--sizes':"[25,50,75,100]",
    defaults={'--fasta':'RF01725.fa','--sizes':"[25,50]",
              '--njobs':'4','--repeats': '5','--debug':'False'}

    for k,v in defaults.items():
        if k not in optlist:
            optlist[k]=v

    for k in ['--sizes','--repeats','--njobs', '--debug']:
        optlist[k] = eval(optlist[k])

    #{'fastafile': 'RF01725.fa', 'mininterfacecount': 2, 'burnin': 4, 'acc_min_sim': 0.24449402485485644, 'imp_lin_start': 0.19892265815047983, 'maxsizediff': 6,
    #'imp_thresh': 0.32120431812249317, 'mincipcount': 2, 'SCORE': -0.0, 'core_choice': False, 'n_samples': 10, 'n_steps': 25, 'quick_skip': True}

    # go over each line of input and start dropping picz
    for nth, line in enumerate(sys.stdin.readlines()):
        if line.strip():
            inputdict=eval(line)
            inputdict.update(optlist)
            r=  get_results(inputdict)
            print( optlist['--sizes'],",",r[0],',',r[1])

