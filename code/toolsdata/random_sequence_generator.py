import random 
num_seqs = 400
minlen = 74
maxlen = 90
samples='RF00005.fa'
symbols='AUGC'

# analyse original
from eden.converter.fasta import fasta_to_sequence
# count all the symbols
symboldict={symbol:0 for symbol in symbols }
for name,seq in fasta_to_sequence(samples):
     for symbol in symboldict.keys():
         symboldict[symbol]+=seq.count(symbol)


def choosesymbol(total,weights,symbols):
    i=random.randint(0,total)
    for e,w in enumerate(weights):
        i-=w
        if i<=0:
            return symbols[e]
    print 'ERRER this should not happen. this means my code sucks'


def make_random_sequence(minlen,maxlen, weights,symbols,total):
    length = random.randint(minlen, maxlen)
    seq = [ choosesymbol(total,weights,symbols) for i in xrange(length) ]
    return ''.join(seq)

 
fasta=''
symbols=symboldict.keys()
weights= [symboldict[k] for k in symbols]
valuesum=sum(symboldict.values())

for i in xrange(num_seqs):
     fasta+='>seq%d\n' %i
     seq=make_random_sequence(minlen,maxlen, weights,symbols,valuesum)
     while seq:
         fasta+=seq[:60]+"\n"
         seq=seq[60:]
print fasta
