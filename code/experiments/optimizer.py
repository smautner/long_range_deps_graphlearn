import Valium.sumsim as ss
import Valium.randset as rs
import utils

def experiment(argz):
    try:
        a, b = utils.get_seq_tups(argz['--fasta'], size=argz['--size'], sizeb=1)
        b = utils.fit_sample(a, arguments=argz)
        a, b = ss.unpack(a, b)
        dist,sim = ss.get_dist_and_sim_crossval(a, b)
        return dist,sim
    except:
        return -1,-1


def repeat_experiment(argz,num):
    scores = [experiment(argz) for i in range(num)]
    dists,sims = utils.transpose(scores)
    # print scores
    return dists[num/2],sims[num/2]


def main(additional_args):
    argz = rs.get_random_params()
    argz.update(additional_args)
    dist,sim = repeat_experiment(argz, num=argz['--repeats'])
    argz['dist'] = dist
    argz['sim']=sim
    return str(argz)


import getopt
import sys
if __name__ == '__main__':    # PARSE THE ARGS:
    optlist, args = getopt.getopt(sys.argv[2:], '', ['fasta=', 'size=', 'repeats='])
    optlist=dict(optlist)
    if '--rfam' not in optlist:
        optlist['--fasta'] = "RF01725.fa"
    if '--size' not in optlist:
        optlist['--size'] = "50"
    if '--repeats' not in optlist:
        optlist['--repeats'] = '3'
    for k in ['--repeats','--size']:
        optlist[k] = int(optlist[k])
    print(main(optlist))



