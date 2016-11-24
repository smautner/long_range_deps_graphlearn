#!/usr/bin/python


# this is what i get from my documentation regarding the drawing
# paste res1 res2 > draw.py infernal
# res1 > draw distribution


import numpy as np
import matplotlib.pyplot as plt


def make_infernal_plot(saveas=0, labels=('G1', 'G2', 'G3', 'G4', 'G5'), means=[(20, 35), (20, 85)],
                       stds=[(2, 3), (3, 3)], labelz=['','']):
    # N = len(labels)
    # ind = np.arange(N)
    # width = 0.35
    plt.figure(figsize=(14, 5))
    fig, ax = plt.subplots()
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(14)
    # ax.ylim(0.0,100)
    plt.ylim(0, 80)
    plt.xlim(0, 400)
    plt.axhline(y=38, color='black', linewidth=3)

    def fillthing(y, std, label='some label', col='b'):
        y = np.array(y)
        std = np.array(std)
        ax.fill_between(labels, y + std, y - std, facecolor=col, alpha=0.3, linewidth=0)
        # ax.plot(labels,y,label=label,color='gray')
        ax.plot(labels, y, color='gray')

    # ax.errorbar(labels, means[0], yerr= stds[0], fmt='o')
    # ax.errorbar(labels, means[1], yerr= stds[1], fmt='x')
    fillthing(means[0], stds[0], col='#6A9AE2')
    # fillthing(means[1],stds[1],col='#8DDD82')
    fillthing(means[1], stds[1], col='#F94D4D')

    ax.plot(labels, means[0], label=labelz[0], color='b', linewidth=2.0)
    ax.plot(labels, means[1], label=labelz[1], color='r', linewidth=2.0)
    # add some text for labels, title and axes ticks
    labelfs = 16
    ax.set_ylabel('Infernal bit score', fontsize=labelfs)
    ax.set_xlabel('training sequences', fontsize=labelfs)
    ax.legend(loc='lower right')

    plt.savefig('%d_infplot.png' % saveas)


def make_dis_plot(saveas, numgraphs, distribution, similarity):
    """
    """
    rc = {'color': 'r'}
    bc = {'color': 'b'}
    ws = 2.2
    o = np.mean(distribution, axis=1)
    o = np.median(distribution, axis=1)
    s = np.mean(similarity, axis=1)
    s = np.median(similarity, axis=1)
    plt.figure(figsize=(18, 8))
    marksize = 5

    # fontsize...
    fsa = 13
    fsb = 15

    # OKOK NEW STUFF TESTING
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    for label in (ax1.get_xticklabels() + ax1.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(fsa)
    for label in (ax2.get_xticklabels() + ax2.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(fsa)
    numgraphs = np.array(numgraphs)

    # plt.grid()
    ax1.set_axisbelow(True)
    ax1.yaxis.grid(color='gray', linestyle='dashed')

    ax1.boxplot(distribution, positions=numgraphs, widths=ws, capprops=bc, medianprops=bc, boxprops=bc, whiskerprops=bc,
                flierprops=None)
    ax2.boxplot(similarity, positions=numgraphs, widths=ws, capprops=rc, medianprops=rc, boxprops=rc, whiskerprops=rc,
                flierprops=None)
    ax1.plot(numgraphs, o, color='b', marker='o', markeredgewidth=1, markersize=marksize, markeredgecolor='b',
             markerfacecolor='w', label='KL divergence', linewidth=2)
    ax2.plot(numgraphs, s, color='r', marker='o', markeredgewidth=1, markersize=marksize, markeredgecolor='r',
             markerfacecolor='w', label='similarity', linewidth=2)

    # plt.xlim(percentages[0]-.05,percentages[-1]+.05)
    print numgraphs
    plt.xlim(min(numgraphs) - 2, max(numgraphs) + 2)
    ax1.set_ylim(0.1, .8)
    ax2.set_ylim(1, 1.05)
    ax1.set_xlim(20, 105)
    ax2.set_xlim(20, 105)
    plt.xticks(numgraphs, numgraphs)

    # plt.title(run_id + '\n', fontsize=18)
    ax1.legend(loc='lower left', fontsize=fsa)
    ax2.legend(loc='upper right', fontsize=fsa)
    # plt.ylabel('ROC AUC',fontsize=18)
    ax1.set_ylabel('divergence', fontsize=fsb)
    ax2.set_ylabel('similarity', fontsize=fsb)
    ax2.set_xlabel('training sequences', fontsize=fsb)
    ax1.set_xlabel('training sequences', fontsize=fsb)
    plt.savefig('%d_displot.png' % saveas)


import sys

if __name__ == "__main__":

    # select the mode
    inf = 'infernal' in sys.argv[1:]
    dis = 'distribution' in sys.argv[1:]
    if inf == dis:
        print "draw needs infernal or distribution as argument"
        exit()
    if inf:
        mode = 'inf'
    else:
        mode = 'dis'

    # go over each line of input and start dropping picz
    for nth, line in enumerate(sys.stdin.readlines()):
        if line.strip():
            if mode == 'inf':
                size, means, stds, void, means2, stds2 = eval(line)
                # [10, 20, 50, 100, 200, 400],[56.725000000000001, 61.393103448275852, 56.91238095238095, 52.69766355140186, 47.822811059907835,48.007709750566889],[13.075456463481306, 12.052927274560741, 12.772826030571943, 14.369074946790544, 15.520753474342225,14.701883300962383],[10, 20, 50, 100, 200, 400],[11.473076923076924, 10.367272727272727, 14.535714285714288, 12.277238805970152, 13.774444444444445,14.713135985198891],[15.433486320336051, 14.398291039456005, 16.261699399107673, 13.963606495020159, 14.966687399223829,14.970472924155272]
                make_infernal_plot(nth, size, [means, means2], [stds, stds2],labelz=['Infernal cmemit','Extended grammar'])
            if mode == 'dis':
                numgraph, distr, simi = eval(line)
                print numgraph, distr, simi
                make_dis_plot(nth, numgraph, distr, simi)
