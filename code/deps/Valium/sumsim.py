##########3
# first implementations.. pretty crap and unused now :) 
########

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

def unpack(a,b):
    a = [aa for x,aa in a]
    b = [bb for x,bb in b]
    return a,b
import math
def calcsimset(a,b):

    ab=simsum(a,b,False)
    aa=simsum(a,a,True) 
    bb=simsum(b,b,True)

    avgab =  ab/(len(a)*len(b))
    avgbb =  bb/(len(b)*len(b)-len(b)) 
    avgaa =  aa/(len(a)*len(a)-len(a))
    arg=avgab/math.sqrt(avgaa*avgbb)
    return arg

if debug:
    s1=['asdasd','asdasd','abc','sdf','xcv']
    s2=['asdasf','asdasd','abc','sdf','xcv']
    print 'testing simsum eq',simsum(s1,s1,True)
    print 'testing simsum neq',simsum(s2,s1)
    print 'testing sim eq', sim(s1[0],s1[0])
    print 'testing sim dif', sim(s2[0],s1[0])
    print calcsimset(s1,s2)


####
# USE EDEN FOR COMPARIZON
###
import eden.path as ep
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math 
def vectorize(a,b):
    v=ep.Vectorizer()
    a = v.transform(a)
    b = v.transform(b)
    return a,b

def similarity_mean(a,b,keepdiag=True):
    simmatrix = cosine_similarity(a,b)
    if not keepdiag:
        pass
        #np.fill_diagonal(simmatrix,0)
    #print 'cosine sim %f \n' % np.mean(simmatrix),simmatrix
    return np.mean(simmatrix)

def simset(a,b):
    return similarity_mean(a,b)/math.sqrt(similarity_mean(a,a,False)*similarity_mean(b,b,False))


'''
s1=['asdasd','asdasd','abc','sdf','xcv','asd','adcxzx']
s2=['asdasd','asdasd','abc','sdf','xcv']
s3=['asfasd','cvxcvxc','werwttwe','wertwet','weryuii']

print 'asd',simset(s1,s2)
print 'asd',simset(s1,s3)
print 'asd',simset(s3,s3)
'''



# ok letz create the new thing to compare the distributions

#######################
#  distrubution comparison
#####################



##############
# first we need an estimator, /
#########
from eden.util import fit_estimator as eden_fit_estimator
from sklearn.calibration import CalibratedClassifierCV
import numpy
from scipy.sparse import vstack
from sklearn.linear_model import SGDClassifier

class OneClassEstimator:
    '''
    there might be a bug connected to nx.digraph..
    '''

    def __init__(self, nu=.5, cv=2, n_jobs=-1,
        classifier=SGDClassifier(loss='log')):
        '''
        Parameters
        ----------
        nu: part of graphs that will be placed in the negative set (0~1)
        cv:
        n_jobs: jobs for fitting
        move_bias_calibrate: after moving the bias we can recalibrate
        classifier: calssifier object
        Returns
        -------
        '''
        self.nu = nu
        self.cv = cv
        self.n_jobs = n_jobs
        self.classifier = classifier



    def fit(self, data_matrix, random_state=None):

        if random_state is not None:
            random.seed(random_state)

        # use eden to fitoooOoO
        self.estimator = self.fit_estimator(data_matrix, n_jobs=self.n_jobs, cv=self.cv, random_state=random_state)

        # move bias to obtain oneclassestimator
        # jaaa we have to work on this...
        self.cal_estimator = self.move_bias(data_matrix, estimator=self.estimator, nu=self.nu, cv=self.cv)

        return self

    def fit_estimator(self, data_matrix, n_jobs=-1, cv=2, random_state=42):
        '''
        create self.estimator...
        by inversing the data_matrix set to get a negative set
        and then using edens fit_estimator
        '''
        # make negative set 
        data_matrix_neg = data_matrix.multiply(-1)


        return eden_fit_estimator(self.classifier, positive_data_matrix=data_matrix,
                                  negative_data_matrix=data_matrix_neg,
                                  cv=cv,
                                  n_jobs=n_jobs,
                                  n_iter_search=10,
                                  random_state=random_state)

    def move_bias(self, data_matrix, estimator=None, nu=.5, cv=2):
        '''
            move bias until nu of data_matrix are in the negative class
            then use scikits calibrate to calibrate self.estimator around the input
        '''
        #  move bias
        # l = [(estimator.decision_function(g)[0], g) for g in data_matrix]
        # l.sort(key=lambda x: x[0])
        # element = int(len(l) * nu)
        # estimator.intercept_ -= l[element][0]

        scores = [estimator.decision_function(sparse_vector)[0]
                  for sparse_vector in data_matrix]
        scores_sorted = sorted(scores)
        pivot = scores_sorted[int(len(scores_sorted) * self.nu)]
        estimator.intercept_ -= pivot

        # calibrate
        if True:
            # data_matrix_binary = vstack([a[1] for a in l])
            # data_y = numpy.asarray([0] * element + [1] * (len(l) - element))
            data_y = numpy.asarray([1 if score >= pivot else -1 for score in scores])
            self.testimator = SGDClassifier(loss='log')
            self.testimator.fit(data_matrix, data_y)
            # estimator = CalibratedClassifierCV(estimator, cv=cv, method='sigmoid')
            estimator = CalibratedClassifierCV(self.testimator, cv=cv, method='sigmoid')
            estimator.fit(data_matrix, data_y)
        return estimator

    def predict_single(self, vectorized_graph):
        if True:
            result = self.cal_estimator.predict_proba(vectorized_graph)[0, 1]
        else:
            result = self.cal_estimator.decision_function(vectorized_graph)[0]

        if False:
            return 1 - result
        return result

    # probably broken ... you should use predict single now o OO
    def predict(self, things):
        return self.predict_single(things)
        #return numpy.array([1 if self.predict_single(thing) > .5 else 0 for thing in things])


def compdistr(a,b):
    """k
    a and b are eden vectors. 
    we train a oneclasssvm on each.
    then we union the vectors.
    for each we get 2 scores, we sum the difference.
    """

    
    e1= OneClassEstimator(n_jobs=1)
    e1.fit(a)
    e2 =OneClassEstimator(n_jobs=1)
    e2.fit(b)

    data=vstack((a,b))
    scoresum=0.0
    for vec in data:
        score1=e1.predict(vec)
        score2=e2.predict(vec)
        scorediff = abs(score1-score2)
        scoresum+=scorediff
        #print score1,score2
    scoreaverage= scoresum / np.shape(data)[0]
    return scoreaverage
  


def score(alist,blist):
    a,b=vectorize(alist,blist)
    distri = compdistr(a,b)
    similarity = simset(a,b)
    #print distri, similarity
    return distri - similarity

def get_similarity(alist,blist):
    a,b=vectorize(alist,blist)
    #distri = compdistr(a,b)
    similarity = simset(a,b)
    return similarity



#### WRKING COMPDISTR
#v=ep.Vectorizer()
#a,b = vectorize(s1,s3,v)
#print compdistr(a,b)
