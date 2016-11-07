


def linereader(f,res):
    for line in f:
        line=line.strip()
        
        # empty lines
        if not line:
            continue

        # new rfam number:
        elif line.startswith('#=GF AC   RF'):
            cdict={}
            res.append(cdict)
            cdict['name']=line[12:]
            cdict['slens']=[]

        # new consensus structure:
        elif line.startswith('#=GF SQ   '):
            cdict['numseq']=int(line[10:])    

        #count
        elif line.startswith('#=GC SS_cons'):
            cdict['structure']=sum(line.count(sym) for sym in ['(',')','<','>'])

        # regular comment
        elif line.startswith('#'):
            continue
        else:
            seq=line[25:]
            nuccnt=len(seq)-seq.count('-')
            cdict['slens'].append(nuccnt)
    for d in res:
        d['slens'] = sum(d['slens']) / len(d['slens'])
        d['name'] = "http://rfam.xfam.org/family/RF%s#tabview=tab3" % d['name']


res=[]
#with open('Rfam.seed','r') as f:
with open('Rfam.seed','r') as f:
    linereader(f,res)
    

res= [d for d in res if d['numseq']> 100 and d['structure']> 40]
res.sort(key=lambda x: x['slens'])

for e in res:
    print e
        
