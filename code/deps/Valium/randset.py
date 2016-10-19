




# soso we need random parametersets
from sklearn.model_selection import ParameterSampler
from scipy.stats.distributions import uniform,randint
import numpy as np
np.random.seed()


#uniform(loc=4,scale=2) # default is something something
# not sure what this does...
#rounded_list = [dict((k, round(v, 6)) for (k, v) in d.items())
#                 for d in param_list]

defaultarg={}
defaultarg['imp_thresh']=uniform()
defaultarg['imp_lin_start']=uniform()
defaultarg['maxsizediff']=randint(low=5,high=20)
defaultarg['acc_min_sim']=uniform(loc=.2,scale=.6)
defaultarg['n_samples']=randint(low=2,high=8) # this many creations PER INSTANCE
defaultarg['n_steps']=randint(low=10,high=100)
defaultarg['quick_skip']=[True,False]
defaultarg['core_choice']=[True,False]
defaultarg['burnin']=randint(low=0, high=15)
defaultarg['mincipcount']=[1,2]#randint(low=1,high=4)
defaultarg['mininterfacecount']=[1,2]#randint(low=1,high=4)




def swapifsmaler(parm,a,b):
    if parm[a] < parm[b]:
        parm[a],parm[b]=parm[b],parm[a]
    return parm

def get_random_params():
    parm=list(ParameterSampler(defaultarg, n_iter=1))[0]
    parm=swapifsmaler(parm,'imp_thresh','imp_lin_start')
    if parm['n_steps'] < parm['burnin']+parm['n_samples']:
        return get_random_params()
    return parm
