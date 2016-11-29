


with open('Snakefile','r') as f:
    text=f.read()

def delicheck(s):
    return s.startswith('"') or s.startswith("'")
def delidel(s):
    return s[1:-1]

def formator(s):
    lines=s.split('\n')
    res=''
    for line in lines:
        line=line.strip()
        while delicheck(line):
            line=delidel(line)
        res+= line
    return res

for rule in text.split('\nrule'):
    lines=rule.split(":")
    lines=map(lambda x: x.strip(), lines)
    name=lines[0]
    print '\n\n---------------------------------------\n'
    #for e in lines: print "'%s'" % e
    for i,line in enumerate(lines):
        if 'shell' in line:
            break
    else:
        print 'didnt find shellcmd'
        continue

    with open(name+".sh",'w') as f:
        f.write( formator( lines[i+1]) )




