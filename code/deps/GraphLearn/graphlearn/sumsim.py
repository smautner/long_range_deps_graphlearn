debug=False

def levenshtein(s1, s2):
    if len(s1) < len(s2):
        return levenshtein(s2, s1)
    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def sim(s1,s2):
    l = float( max(len(s1),len(s2))) 
    lp = l - float(levenshtein(s1,s2))
    return lp/l 


def simsum(a,b,del_diag=False):
    res=0.0
    for i, ea in enumerate(a):
        for j, eb in enumerate(b):
            if del_diag and i==j:
                continue
            res+=sim(ea,eb)
    return res

import math
def calcsimset(a,b):
    print a,b
    a = [aa for x,aa in a]
    b = [bb for x,bb in b]
    ab=simsum(a,b,False)
    #aa=simsum(a,a,False) 
    #bb=simsum(b,b,False)
    return  ab/(len(a)*len(b))
    '''
    cc=aa*bb
    if cc==0:
        return ab

    if debug:
        print cc,ab,aa,bb
        print ab/math.sqrt(cc)
    return ab/math.sqrt(cc)
    '''

if debug:
    s1=['asdasd','asdasd','abc']
    s2=['zxczxc','asdasd','abc']
    print 'testing simsum eq',simsum(s1,s1,True)
    print 'testing simsum neq',simsum(s2,s1)
    print 'testing sim eq', sim(s1[0],s1[0])
    print 'testing sim dif', sim(s2[0],s1[0])
    calcsimset(s1,s2)

