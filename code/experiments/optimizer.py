



import Valium.sumsim as ss
import Valium.randset as rs
import utils

def run_and_score(argz):
    # print  "STARTED A RUN" # THIS IS THE NU DEBUG
    try:
        a, b = utils.get_seq_tups(argz['fasta'], size=argz['sizea'], sizeb=argz['sizeb'])
        b = utils.fit_sample(a, arguments=argz)
        a, b = ss.unpack(a, b)
        print "generated_seqs %d" % len(b)
        score = ss.score(a, b)
    except:
        print '.'
        return run_and_score(argz)
    return score


def meaning(argz, num=9):
    scores = [run_and_score(argz) for i in range(num)]
    scores.sort()
    # print scores
    return scores[num / 2]


def zeloop():
    currenthigh = -2
    while True:
        argz = rs.get_random_params()
        argz['fasta']='RF01725.fa'
        argz['sizea']=50
        argz['sizeb']=100

        res = meaning(argz, num=9)
        print res
        if currenthigh < res:
            currenthigh = res
            print '\n' + str(argz)


zeloop()
