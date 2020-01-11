# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 13:00:05 2019

@author: rlwood08
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
from matplotlib import colors

'''You need to change the path to the folder where you saved the NPY files.
    You can also change the error rate and number of reads to the desired
    combination '''
path = 'C:/Usr/NPY/'
error_rate = 10
num_of_reads = 10000


plt.rc('xtick',labelsize=22)
plt.rc('ytick',labelsize=22)
def plot_conf_matrix(conf, classes, save_title, fmt='.3f', cmap=plt.cm.Blues):

    norm = colors.Normalize(vmin=0, vmax=1)

    fig = plt.figure(figsize=(18,17))
    gs = grd.GridSpec(2,7, figure=fig, width_ratios=[25,25,25,25,25,25,3])
    ax1 = fig.add_subplot(gs[0,:2])
    ax1.imshow(conf[0], norm=norm, interpolation='nearest', cmap=cmap)
    ax1.set(xticks=np.arange(conf[0].shape[1]),
                yticks=np.arange(conf[0].shape[0]),
                xticklabels=classes[:conf[0].shape[1]], yticklabels=classes)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
    ax1.set_title('ET',fontdict={'fontsize':24})
    
    ax2 = fig.add_subplot(gs[0,2:4])
    ax2.imshow(conf[1], norm=norm, interpolation='nearest', cmap=cmap)
    ax2.set(xticks=np.arange(conf[0].shape[1]),
                xticklabels=classes[:conf[0].shape[1]])
    # Rotate the tick labels and set their alignment.
    plt.setp(ax2.get_yticklabels(), visible=False)    
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
    ax2.yaxis.set_tick_params(which='both',length=0)
    ax2.set_title('NN',fontdict={'fontsize':24})
    
    ax3 = fig.add_subplot(gs[0,4:6])
    ax3.imshow(conf[2], norm=norm, interpolation='nearest', cmap=cmap)
    ax3.set(xticks=np.arange(conf[0].shape[1]),
                xticklabels=classes[:conf[0].shape[1]])
    # Rotate the tick labels and set their alignment.
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.setp(ax3.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
    ax3.yaxis.set_tick_params(which='both',length=0)
    ax3.set_title('LDA',fontdict={'fontsize':24})
    
    ax4 = fig.add_subplot(gs[1,1:3])
    ax4.imshow(conf[3], norm=norm, interpolation='nearest', cmap=cmap)
    ax4.set(xticks=np.arange(conf[0].shape[1]),
                yticks=np.arange(conf[0].shape[0]),
                xticklabels=classes[:conf[0].shape[1]], yticklabels=classes)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax4.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
    ax4.set_title('PA',fontdict={'fontsize':24})
    
    ax5 = fig.add_subplot(gs[1,3:5])
    im5 = ax5.imshow(conf[4], norm=norm, interpolation='nearest', cmap=cmap) 
    ax5.set(xticks=np.arange(conf[0].shape[1]),
                xticklabels=classes[:conf[0].shape[1]])
    # Rotate the tick labels and set their alignment.
    plt.setp(ax5.get_yticklabels(), visible=False)
    plt.setp(ax5.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")
    ax5.yaxis.set_tick_params(which='both',length=0)
    ax5.set_title('GNB',fontdict={'fontsize':24})
    
    ax6 = fig.add_subplot(gs[:,6])
    ax6.figure.colorbar(im5, cax=ax6)

    # We want to show all ticks...
    for axe in [ax1,ax2,ax3,ax4,ax5]:
        axe.set(xticks=np.arange(conf[0].shape[1]),
                yticks=np.arange(conf[0].shape[0]),
                xticklabels=classes[:conf[0].shape[1]], yticklabels=classes)

        # Rotate the tick labels and set their alignment.
        plt.setp(axe.get_xticklabels(), rotation=45, ha="right",
                rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(conf[0].shape[0]):
        for j in range(conf[0].shape[1]):
            for int_ax,axe in enumerate([ax1,ax2,ax3,ax4,ax5]): 
                if conf[int_ax][i, j] <= 0.005:
                    axe.text(j, i, 0,
                            ha="center", va="center",
                            color="white" if conf[int_ax][i, j] > 0.45 else "black", fontsize=20)
                elif conf[int_ax][i, j] >= 0.995:
                    axe.text(j, i, 1,
                            ha="center", va="center",
                            color="white" if conf[int_ax][i, j] > 0.45 else "black", fontsize=20)
                else:
                    axe.text(j, i, format(conf[int_ax][i, j], fmt),
                            ha="center", va="center",
                            color="white" if conf[int_ax][i, j] > 0.45 else "black", fontsize=20)

    fig.text(0.02,.5,'True Label',fontsize=30,rotation='vertical')
    fig.text(.4,0.02,'Predicted Label', fontsize=30)

    #plt.tight_layout()
    plt.savefig('%s.png' % (save_title), dpi='figure', bbox_inches='tight')
    #plt.savefig(os.path.join('J:/groups/dnafingers/Fingerprints Analysis/Graphics/EPS/','%s.eps' % (save_title)), dpi='figure', bbox_inches='tight')
    plt.close()
    return


g_ET = np.zeros((12,10))
g_NN = np.zeros((12,10))
g_LDA = np.zeros((12,10))
g_PA = np.zeros((12,10))
g_GNB = np.zeros((12,10))
p_ET = np.zeros((5,5))
p_NN = np.zeros((5,5))
p_LDA = np.zeros((5,5))
p_PA = np.zeros((5,5))
p_GNB = np.zeros((5,5))
pg_ET = np.zeros((2,2))
pg_NN = np.zeros((2,2))
pg_LDA = np.zeros((2,2))
pg_PA = np.zeros((2,2))
pg_GNB = np.zeros((2,2))
for ii in range(10):
    g_ET[:10,:] += np.load('%s%s %s ET Genome Confusion Matrix Test %s.npy' %(path,error_rate,num_of_reads,ii)) 
    g_NN[:10,:] += np.load('%s%s %s Neural Network Genome Confusion Matrix Test %s.npy' %(path,error_rate,num_of_reads,ii))
    g_PA[:10,:] += np.load('%s%s %s Passive-Aggressive Genome Confusion Matrix Test %s.npy' %(path,error_rate,num_of_reads,ii))
    g_GNB[:10,:] += np.load('%s%s %s GNB Genome Confusion Matrix Test %s.npy' %(path,error_rate,num_of_reads,ii))
    g_LDA[:10,:] += np.load('%s%s %s LDA Genome Confusion Matrix Test %s.npy' %(path,error_rate,num_of_reads,ii))
    p_ET += np.load('%s%s %s ET Plasmid Confusion Matrix Test %s.npy' %(path,error_rate,num_of_reads,ii))
    p_NN += np.load('%s%s %s Neural Network Plasmid Confusion Matrix Test %s.npy' %(path,error_rate,num_of_reads,ii))
    p_PA += np.load('%s%s %s Passive-Aggressive Plasmid Confusion Matrix Test %s.npy' %(path,error_rate,num_of_reads,ii))
    p_GNB += np.load('%s%s %s GNB Plasmid Confusion Matrix Test %s.npy' %(path,error_rate,num_of_reads,ii))
    pg_ET += np.load('%s%s %s ET Plasmid Group Confusion Matrix Test %s.npy' %(path,error_rate,num_of_reads,ii))
    pg_NN += np.load('%s%s %s Neural Network Plasmid Group Confusion Matrix Test %s.npy' %(path,error_rate,num_of_reads,ii))
    pg_PA += np.load('%s%s %s Passive-Aggressive Plasmid Group Confusion Matrix Test %s.npy' %(path,error_rate,num_of_reads,ii))
    pg_GNB += np.load('%s%s %s GNB Plasmid Group Confusion Matrix Test %s.npy' %(path,error_rate,num_of_reads,ii))
    
pg_LDA += np.load('%s%s %s LDA Plasmid Group Confusion Matrix Test 9.npy'%(path,error_rate,num_of_reads))
p_LDA += np.load('%s%s %s LDA Plasmid Confusion Matrix Test 9.npy'%(path,error_rate,num_of_reads))    
ge_ET = np.zeros((10,10))
ge_NN = np.zeros((10,10))
ge_LDA = np.zeros((10,10))
ge_PA = np.zeros((10,10))
ge_GNB = np.zeros((10,10))
for ii in range(10):
    ge_ET += np.load('%s%s %s ET Genome Extras Confusion Matrix Test %s.npy' %(path,error_rate,num_of_reads,ii)) 
    ge_NN += np.load('%s%s %s Neural Network Genome Extras Confusion Matrix Test %s.npy' %(path,error_rate,num_of_reads,ii))
    ge_PA += np.load('%s%s %s Passive-Aggressive Genome Extras Confusion Matrix Test %s.npy' %(path,error_rate,num_of_reads,ii))
    ge_GNB += np.load('%s%s %s GNB Genome Extras Confusion Matrix Test %s.npy' %(path,error_rate,num_of_reads,ii))
    ge_LDA += np.load('%s%s %s LDA Genome Extras Confusion Matrix Test %s.npy' %(path,error_rate,num_of_reads,ii))

g_ET[11,:] += [0., 0., 0., 10000, 0., 0., 0., 0., 0., 0.]
g_NN[11,:] += [0., 0., 0., 10000, 0., 0., 0., 0., 0., 0.]
g_PA[11,:] += [0., 0., 0., 10000, 0., 0., 0., 0., 0., 0.]
g_GNB[11,:] += [0., 0., 0., 10000, 0., 0., 0., 0., 0., 0.]
g_LDA[11,:] += [0., 0., 0., 10000, 0., 0., 0., 0., 0., 0.]

g_ET[10,:] += ge_ET.sum(axis=0)-g_ET[11,:]
g_NN[10,:] += ge_NN.sum(axis=0)-g_NN[11,:]
g_PA[10,:] += ge_PA.sum(axis=0)-g_PA[11,:]
g_GNB[10,:] += ge_GNB.sum(axis=0)-g_GNB[11,:]
g_LDA[10,:] += ge_LDA.sum(axis=0)-g_LDA[11,:]

g_ET = g_ET.astype('float')/g_ET.sum(axis=1)[:,np.newaxis]
g_NN = g_NN.astype('float')/g_NN.sum(axis=1)[:,np.newaxis]
g_LDA = g_LDA.astype('float')/g_LDA.sum(axis=1)[:,np.newaxis]
g_PA = g_PA.astype('float')/g_PA.sum(axis=1)[:,np.newaxis]
g_GNB = g_GNB.astype('float')/g_GNB.sum(axis=1)[:,np.newaxis]
p_ET = p_ET.astype('float')/p_ET.sum(axis=1)[:,np.newaxis]
p_NN = p_NN.astype('float')/p_NN.sum(axis=1)[:,np.newaxis]
p_LDA = p_LDA.astype('float')/p_LDA.sum(axis=1)[:,np.newaxis]
p_PA = p_PA.astype('float')/p_PA.sum(axis=1)[:,np.newaxis]
p_GNB = p_GNB.astype('float')/p_GNB.sum(axis=1)[:,np.newaxis]
pg_ET = pg_ET.astype('float')/pg_ET.sum(axis=1)[:,np.newaxis]
pg_NN = pg_NN.astype('float')/pg_NN.sum(axis=1)[:,np.newaxis]
pg_LDA = pg_LDA.astype('float')/pg_LDA.sum(axis=1)[:,np.newaxis]
pg_PA = pg_PA.astype('float')/pg_PA.sum(axis=1)[:,np.newaxis]
pg_GNB = pg_GNB.astype('float')/pg_GNB.sum(axis=1)[:,np.newaxis]

g_class = ['B. fragilis','C. jejuni','E. coli','K. pneumoniae','S. enterica','S. aureus','S. pneumoniae','S. pyogenes','E. hirae','E. fergusonii','K. aerogenes','M. tuberculosis']
p_class = ['IMP','KPC','NDM','No Resistance','VIM']
pg_class = ['No Resistance','Resistance']

confg = [g_ET, g_NN, g_LDA, g_PA, g_GNB]
confp = [p_ET, p_NN, p_LDA, p_PA, p_GNB]
confpg = [pg_ET, pg_NN, pg_LDA, pg_PA, pg_GNB]
for it,cl,titl in [(confg,g_class,'Genome Confusion Matrix 10 10000'), (confp,p_class,'Plasmid Confusion Matrix 10 100000'), (confpg,pg_class,'Plasmid Group Confusion Matrix 10 100000')]:
    plot_conf_matrix(it, cl, titl)

# [['true negative', 'false positive'],
#       ['false negative', 'true positive']]
# So anything that is in the same row as a true label
# but not correct is a false negative.
# Which means things in the same column as a predicted label
# but not correct are a false positive.
