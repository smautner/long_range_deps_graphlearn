import os
import subprocess as sb
import base64
import time
import io

__version__ = '0.2'

"""
auto ipython backuper..
call dump.dump(this_noteboook,[git tracked imports],saves_go_here) 

here is the plan:
    - save notebook w/o pics as $NAME.bu
    - commit the thing
    - also commit all the py files in related repos
    - get all commits and write them to the restore.sh
    - commit the restore thing
"""


###############################
# main function 
###############################
def dump(notebook = 'no.ipynb',git=[],savedir='../sav', debug=False):
    

    # arg check
    notebookpath = os.path.abspath(notebook)
    savepath = os.path.abspath(savedir)    
    if os.path.isfile(notebookpath) and os.path.isdir(savepath):
        print 'located backup-dir and notebook'
    else:
        print 'cant locate %s or %s' % (notebookpath, savepath)
        exit()
    notebookname = os.path.basename(notebookpath)
    
    # save notebook
    savfile = os.path.join(savepath,notebookname+".bu.ipynb")
    save_notebook(notebookpath,savfile)
    cmd = 'git add %s ; git commit %s -m "autobackup"' % (savfile,savfile)
    if debug:
        print cmd
        myhas='myhas :)'
    else: 
        excode,error,out = shell_exec(cmd)
        print error
        cmd = 'git log -n 1 --pretty=format:"%H"'
        excode,error,myhas = shell_exec(cmd)
        print error 
    # commit all related gits
    git_repos = map(gitstuff,git) # gitstuff -> (rem, path, hash)
    if debug:
        print git_repos

    # write shellscrypt
    script=''

    # write other git repos
    for rem,path,has in git_repos:
        repname=os.path.basename(rem)
        pathrep = os.path.abspath(repname)
        script+='if [ ! -d %s ]; then git clone %s;fi\n' % (repname,rem)
        script+='cd %s ; git checkout %s; cd .. ; PYTHONPATH="%s:${PYTHONPATH}"\n' % (repname,has,pathrep)    

    # write this git repo
    script+='git checkout %s %s \n' % (myhas,savfile)
    script+='cp %s . \n' % savfile
    script+='jupyter notebook'
    if debug:
        print '\n\n\n',script
    else:
        fname=time.strftime("%Y_%m_%d_%H_%M_%S")+notebookname+'.sh'
        ffname= os.path.join(savepath,fname)
        with open(ffname,'w') as f:
            f.write(script)
        cmd= 'git add %s; git commit %s -m "autolog"' % (ffname,ffname)


###############################
# clean and save a notebook with a new name
#############################3
def save_notebook(notebook,save_here):
    # nimm notebook, remove pics , save as save_here	
    doit(notebook,save_here)
    # i just call the copypasted stuff from ze internet    


import io
try:
    # Jupyter >= 4
    from nbformat import read, write, NO_CONVERT
except ImportError:
    # IPython 3
    try:
        from IPython.nbformat import read, write, NO_CONVERT
    except ImportError:
        # IPython < 3
        from IPython.nbformat import current
    
        def read(f, as_version):
            return current.read(f, 'json')
    
        def write(nb, f):
            return current.write(nb, f, 'json')

def _cells(nb):
    """Yield all cells in an nbformat-insensitive manner"""
    if nb.nbformat < 4:
        for ws in nb.worksheets:
            for cell in ws.cells:
                yield cell
    else:
        for cell in nb.cells:
            yield cell

def strip_output(nb):
    """strip the outputs from a notebook object"""
    nb.metadata.pop('signature', None)
    for cell in _cells(nb):
        if 'outputs' in cell:
            cell['outputs'] = []
        if 'prompt_number' in cell:
            cell['prompt_number'] = None
    return nb

def doit(filename,filename2):
    with io.open(filename, 'r', encoding='utf8') as f:
        nb = read(f, as_version=NO_CONVERT)
    nb = strip_output(nb)
    with io.open(filename2, 'w', encoding='utf8') as f:
        write(nb, f)



#####################################
# GIT
#####################################
def gitstuff(g):
    # get path to repository 
    exec("import %s as asshat" % g)
    path = asshat.__file__
    path = path[:path.rfind('/')]

    # get status, look for changes (in py files)
    cmd= 'cd %s ; git status -s' % path
    res = sb.check_output(cmd,shell=True).split("\n")
    changes=map(modfile,res)
    changes=''.join(changes)
    
    # are there changes? commit!
    if changes:
        cmd= 'cd %s ; git commit %s -m "auto commit"' % (path,changes)
        #print cmd
        shell_exec(cmd)
        print "note to self.. do a commit"    

    # get  hash
    cmd= 'cd %s ; git log -n 1 --pretty=format:"%%H"' % path
    has = sb.check_output(cmd,shell=True)

    # get remote  or path to git
    cmd= 'cd %s ;  git remote -v' % path
    remote = sb.check_output(cmd,shell=True)
    if remote.strip():
        rem = remote.split()[1]
    else:
        rem = None
    return (rem,path,has)


def modfile(fi):
    if len(fi)<6:
        return ''
    if fi[:2]==' M' and fi[-3:]=='.py':
        return fi[2:]
    return ''

###################################
# shell exec ..  sanity wrapper for the terrible subprocess module 
#################################
import subprocess
def shell_exec(cmd): 
    '''
    returns (retcode,stderr, stdout)
    '''
    process = subprocess.Popen(cmd,stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
    output, stderr = process.communicate()
    retcode = process.poll()
    return (retcode,stderr,output)

