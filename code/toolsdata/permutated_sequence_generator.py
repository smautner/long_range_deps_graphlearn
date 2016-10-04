from eden.converter.fasta import fasta_to_sequence
import random 

def permute_sequence(st):
    st=list(st)    
    random.shuffle(st)
    return ''.join(st)

def read_and_permute(samples='RF00005.fa'):
    for name,seq in fasta_to_sequence(samples):
        seq=permute_sequence(seq)
        yield (name,seq)
