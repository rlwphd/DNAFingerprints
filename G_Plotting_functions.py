# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 22:22:02 2019

@author: rlwood08
"""

import os
import json
import time
import datetime
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as grd
from dask.distributed import Client, LocalCluster
import dask.dataframe as ddf
import dask.array as da

'''You need to change the path to the folder where you saved the SERS files.'''
local_SERS = 'C:/Usr/SERS/'


plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.labelweight'] = 'bold'
def plotting_data(color_array, data_array, names, data_range, label_int, title='no title', save='no', normalized_type='Random'):
    
    """ This determines how many points are being plotted and then
        creates the necessary gradients for creating the A,T,C,G
        colorbars that go with the plot. """
    index_values = list(range(1, len(color_array[0]) + 1))
    s = [60 for n in range(len(color_array[0]))]   #the size of the markers in plots
    gradient1 = np.vstack((color_array[0], color_array[0]))
    gradient2 = np.vstack((color_array[1], color_array[1]))    
    gradient3 = np.vstack((color_array[2], color_array[2]))    
    gradient4 = np.vstack((color_array[3], color_array[3]))    
    
    fig = plt.figure(figsize=(20,10))
    
    """ Creates a set of 5 graphs inside the figure for the actual
        plot and then the 4 colorbar plots for representing the 
        ATCG percentages """
    #gs = grd.GridSpec(5, 1, height_ratios=[15,1,1,1,1], hspace=0.05)
    gs = grd.GridSpec(6, 1, height_ratios=[15,0.5,1,1,1,1], hspace=0.05)
    """ Working on formatting the actual plot. Getting rid of the x-axis,
        setting the necessary boundaries for the plot, adding a legend
        and a title (if applicable) """
    ax = plt.subplot(gs[0],xticklabels=data_range,xticks=label_int)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    #ax.xaxis.set_ticks([])
    ax.set_xlim(index_values[0]-1,index_values[-1]+1)
    plt.setp(ax.spines.values(), linewidth=2)
    ax.xaxis.set_tick_params(width=2)
    ax.yaxis.set_tick_params(width=2)
    
    if normalized_type == 'no':
        ax.set_ylabel('10-mer Frequency',fontdict={'size':20})
    else:
#        ax.set_ylim(0,1.01)
        ax.set_ylabel('10-mer Frequency Deviation from Bias',fontdict={'size':20})
        
    # looping through each data set for plotting
    max_data = 0
    min_data = 0
    mark = ['o','o','s','v','*','x']
    color = ['k','tab:orange','tab:purple','tab:cyan','tab:gray','tab:brown','tab:pink']
    for row in range(1,len(data_array)):
        if max(data_array[row]) > max_data:
            max_data = max(data_array[row]) + 0.05*max(data_array[row])
        if min(data_array[row]) < min_data:
            if min(data_array[row]) > 0 : 
                min_data = min(data_array[row]) - 0.05*min(data_array[row])
            else:
                min_data = min(data_array[row]) + 0.05*min(data_array[row])
        ax.set_ylim(min_data, max_data)
        ax.scatter(index_values, data_array[row], s=s, c=color[row], marker=mark[row], label=names[row])
    ax.plot(index_values, data_array[0], 'k-', label=names[0], linewidth=2)

    if not title == 'no title':
        plt.title(title)
    plt.legend(prop={'size':16})
    
    """ Creating the colorbars for visualing representing the ATCG
        percenatages"""
    colorAx1 = plt.subplot(gs[2])
    colorAx1.imshow(gradient1, aspect = 'auto', cmap=plt.get_cmap('Blues'))
    pos = list(colorAx1.get_position().bounds)
    x_text = pos[0] - 0.01
    y_text = pos[1] + pos[3]/2
    fig.text(x_text, y_text, 'C', va='center', ha='center')
    colorAx1.set_axis_off()
    
    colorAx2 = plt.subplot(gs[3])
    colorAx2.imshow(gradient2, aspect = 'auto', cmap=plt.get_cmap('Greens'))
    pos = list(colorAx2.get_position().bounds)
    x_text = pos[0] - 0.01
    y_text = pos[1] + pos[3]/2
    fig.text(x_text, y_text, 'G', va='center', ha='center')
    colorAx2.set_axis_off()
    
    colorAx3 = plt.subplot(gs[4])
    colorAx3.imshow(gradient3, aspect = 'auto', cmap=plt.get_cmap('Reds'))
    pos = list(colorAx3.get_position().bounds)
    x_text = [pos[0] - 0.01, pos[0] + pos[2]*5.5/286, pos[0] + pos[2]*16/286, pos[0] + pos[2]*26/286, pos[0] + pos[2]*36/286, pos[0] + pos[2]*44.75/286, pos[0] + pos[2]*53.5/286, pos[0] + pos[2]*62/286, pos[0] + pos[2]*70/286, pos[0] + pos[2]*78.25/286, pos[0] + pos[2]*86/286, pos[0] + pos[2]*93.5/286, pos[0] + pos[2]*100.5/286, pos[0] + pos[2]*107.75/286, pos[0] + pos[2]*114.75/286, pos[0] + pos[2]*121.5/286, pos[0] + pos[2]*128.25/286, pos[0] + pos[2]*134/286, pos[0] + pos[2]*140/286, pos[0] + pos[2]*146/286, pos[0] + pos[2]*152/286, pos[0] + pos[2]*158/286, pos[0] + pos[2]*163.5/286, pos[0] + pos[2]*168.75/286, pos[0] + pos[2]*173.5/286, pos[0] + pos[2]*178.5/286, pos[0] + pos[2]*183.5/286, pos[0] + pos[2]*188.75/286, pos[0] + pos[2]*193.5/286, pos[0] + pos[2]*198.15/286, pos[0] + pos[2]*202.15/286, pos[0] + pos[2]*206.2/286, pos[0] + pos[2]*209.95/286, pos[0] + pos[2]*214.15/286, pos[0] + pos[2]*218.25/286, pos[0] + pos[2]*222/286, pos[0] + pos[2]*226/286]
    y_text = [pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2]
    label_text = ['T', '0', '1', '0', '2', '1', '0', '3', '2', '1', '0', '4', '3', '2', '1', '0', '5', '4', '3', '2', '1', '0', '6', '5', '4', '3', '2', '1', '0', '7', '6', '5', '4', '3', '2', '1', '0']
    for ii in range(len(label_text)):
        fig.text(x_text[ii], y_text[ii], label_text[ii], va='center', ha='center')
    colorAx3.set_axis_off()
    
    colorAx4 = plt.subplot(gs[5])
    colorAx4.imshow(gradient4, aspect = 'auto', cmap=plt.get_cmap('Greys'))
    pos = list(colorAx4.get_position().bounds)
    x_text = [pos[0] - 0.01, pos[0] + pos[2]*5.5/286, pos[0] + pos[2]*16/286, pos[0] + pos[2]*26/286, pos[0] + pos[2]*36/286, pos[0] + pos[2]*44.75/286, pos[0] + pos[2]*53.5/286, pos[0] + pos[2]*62/286, pos[0] + pos[2]*70/286, pos[0] + pos[2]*78.25/286, pos[0] + pos[2]*86/286, pos[0] + pos[2]*93.5/286, pos[0] + pos[2]*100.5/286, pos[0] + pos[2]*107.75/286, pos[0] + pos[2]*114.75/286, pos[0] + pos[2]*121.5/286, pos[0] + pos[2]*128/286, pos[0] + pos[2]*134/286, pos[0] + pos[2]*140/286, pos[0] + pos[2]*146/286, pos[0] + pos[2]*152/286, pos[0] + pos[2]*158/286, pos[0] + pos[2]*163.5/286, pos[0] + pos[2]*168.5/286, pos[0] + pos[2]*173.5/286, pos[0] + pos[2]*178.5/286, pos[0] + pos[2]*183.5/286, pos[0] + pos[2]*188.75/286, pos[0] + pos[2]*193.5/286, pos[0] + pos[2]*198.15/286, pos[0] + pos[2]*202.15/286, pos[0] + pos[2]*206.2/286, pos[0] + pos[2]*209.95/286, pos[0] + pos[2]*214.15/286, pos[0] + pos[2]*218.25/286, pos[0] + pos[2]*222.05/286, pos[0] + pos[2]*226/286]
    y_text = [pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2, pos[1] + pos[3]/2]
    label_text = ['A', '0', '0', '1', '0', '1', '2', '0', '1', '2', '3', '0', '1', '2', '3', '4', '0', '1', '2', '3', '4', '5', '0', '1', '2', '3', '4', '5', '6', '0', '1', '2', '3', '4', '5', '6', '7']
    for ii in range(len(label_text)):
        fig.text(x_text[ii], y_text[ii], label_text[ii], va='center', ha='center')
    colorAx4.set_axis_off()
    x_text1 = pos[0] + pos[2]/2
    y_text1 = pos[1] - 0.03
    fig.text(x_text1, y_text1, 'Fractional Base Composition Spectrum', va='center', ha='center',fontdict={'size':20})
    
    """ Either showing the plot or just directly saving it """
    if save == 'yes':
        if not normalized_type == 'no':
            plt.savefig(normalized_type + ' ' + title + '.png', dpi='figure', bbox_inches='tight')
            plt.close()
        else:
            plt.savefig('Not Normalized ' + title + '.png', dpi='figure', bbox_inches='tight')
            plt.close()
    else:
        plt.show()

    return

start = time.perf_counter()

# creating the categorical labels for storing data
dna_length = 10
# creating the correct tuples for how many A, T, G and C's are in each bin
kmer_range = [(aa, tt, gg, cc) for aa in range(dna_length + 1) for tt in range(dna_length + 1) for gg in range(dna_length + 1) for cc in range(dna_length + 1) if aa + tt + cc + gg == dna_length]
# setting dna length based bias
bias = np.array([(1/4**dna_length) * math.factorial(dna_length)/(math.factorial(kmer[0]) * math.factorial(kmer[1]) * math.factorial(kmer[2]) * math.factorial(kmer[3])) for kmer in kmer_range])
# creating the categorical labels for storing data
data_categories = ["A%sT%sG%sC%s" % (str(aa), str(tt), str(gg), str(cc)) for aa in range(dna_length + 1) for tt in range(dna_length + 1) for gg in range(dna_length + 1) for cc in range(dna_length + 1) if aa + tt + cc + gg == dna_length]
# creating the labels for the non-numerical information
data_index = ['Seq Record ID', 'Resistance', 'Name', 'Genus', 'DNA Type', 'Strain', 'Bacteria Type', 'Notes']
# combining the the non-numerical and categorical labels for the pandas dataframe
data_index.extend(data_categories)

# Number of samples per species 
num_training_samples = 1000
# List of error testing rates
error_rate = [0]
# List of number of optical sequencing reads
num_reads = [1000000]

# Getting the list of all the data files that need to be tested
file_list = [file for file in os.listdir(local_SERS)]
    
# Intializing dask to run things in parallel
with LocalCluster(processes=False) as cluster, Client(cluster) as client:
    
# Cycling through the 24 combinations of error rate and number of reads
    for error in error_rate:
    # Recording the time it takes to run everything
        for reads in num_reads:
        # Getting the error and reads values in string form for file identification
            str_err = '_%s_' % (int(100*error))
            str_read = '_%s_' % (reads)
        # Getting the list of the files for the specific reads and errors
            working_genome_list = [file for file in file_list if str_err in file and str_read in file and 'Genome' in file]
        # Making sure that the lists aren't empty
            run_list = []    
            if len(working_genome_list) != 0:
                run_list.append(working_genome_list)
        
            for use_list in run_list:
            # Retrieving wether the file is genomic or plasmid
                file_type = 'Genome'
    
            # Determining how long this takes
                tick = time.perf_counter()
            # Splitting the files into the unseen sets and the training sets
                train_set = []
                unseen_test_set = []
                check_set = []
                file_info = []
            # Getting each file in the list of files to use
                for file in use_list:
                    if file.split('_')[3] == 'Training':
                        train_file = file
                    elif file.split('_')[3] == 'Testing':
                        test_file = file
                    elif file.split('_')[3] == 'Extras':
                        extra_file = file
            # Opening up each of the HDF5 files in the working list
                store_train = ddf.read_hdf(os.path.join(local_SERS,train_file), 'df*')
                store_test = ddf.read_hdf(os.path.join(local_SERS,test_file), 'df*')
                store_extra = ddf.read_hdf(os.path.join(local_SERS,extra_file), 'df*')
            # Storing just the values of the data split
                train_set.append(store_train.drop('Name',axis='columns').to_dask_array(True))
                unseen_test_set.append(store_test.drop('Name',axis='columns').to_dask_array(True))
                check_set.append(store_extra.drop('Name',axis='columns').to_dask_array(True))
                bac_train = np.char.array(store_train.loc[:,'Name'].compute().to_list())
                bac_test = np.char.array(store_test.loc[:,'Name'].compute().to_list())
                bac_ex = np.char.array(store_extra.loc[:,'Name'].compute().to_list())
                bac_train = np.where(bac_train.rfind('_') != bac_train.find('_'), bac_train.rpartition('_')[:,0].replace('_',' '),bac_train.replace('_',' '))
                bac_test = np.where(bac_test.rfind('_') != bac_test.find('_'), bac_test.rpartition('_')[:,0].replace('_',' '),bac_test.replace('_',' '))
                bac_ex = np.where(bac_ex.rfind('_') != bac_ex.find('_'), bac_ex.rpartition('_')[:,0].replace('_',' '),bac_ex.replace('_',' '))
                bac_ex = np.where(bac_ex == 'Enterobacter aerogenes', 'Klebsiella aerogenes', bac_ex)
            # Creating a testing data array for the unseen tests and the training data
                train_data = da.concatenate([arr[int(jj*num_training_samples):int(1+jj*num_training_samples)] for arr in train_set for jj in range(len(arr.chunks[0]))],axis=0).compute()
                unseen_data = da.concatenate([arr[int(jj*num_training_samples):int(1+jj*num_training_samples)] for arr in unseen_test_set for jj in range(len(arr.chunks[0]))],axis=0).compute()
                check_data = da.concatenate([arr[int(jj*num_training_samples):int(1+jj*num_training_samples)] for arr in check_set for jj in range(len(arr.chunks[0]))],axis=0).compute()
            # Creating the labels/categories for the data
                train_names = da.concatenate([np.array(bac_train[int(jj*num_training_samples):int(1+jj*num_training_samples)]) for arr in train_set for jj in range(len(arr.chunks[0]))],axis=0).compute()
                unseen_names = da.concatenate([np.array(bac_test[int(jj*num_training_samples):int(1+jj*num_training_samples)]) for arr in unseen_test_set for jj in range(len(arr.chunks[0]))],axis=0).compute()
                check_names = da.concatenate([np.array(bac_ex[int(jj*num_training_samples):int(1+jj*num_training_samples)]) for arr in check_set for jj in range(len(arr.chunks[0]))],axis=0).compute()

                data = np.concatenate((train_data,unseen_data,check_data,np.zeros((1,286))))
                df = pd.DataFrame(data=data,columns=data_categories)
                
                pdf = df.iloc[[22,3,17,20],:]
                p2df = df.iloc[[22,3,17,20],:]
                p2df += bias
                pnames = ['Bias',train_names[3],unseen_names[7],check_names[0]]
                
                with open('Split_data_range.txt', 'r') as f:
                    data_range = json.load(f)
                
                with open('Split color array 10mer data.txt', 'r') as f:
                    color_array = json.load(f)

                bplot = pdf[data_range].to_numpy()
                nplot = p2df[data_range].to_numpy()
                
                plotting_data(color_array,bplot, pnames, [data_range[0],data_range[44],data_range[77],data_range[107],data_range[146],data_range[178],data_range[210],data_range[241],data_range[285]], [1,45,78,108,147,179,211,242,286], save='no', normalized_type='Split')
                plotting_data(color_array, nplot, pnames, [data_range[0],data_range[44],data_range[77],data_range[107],data_range[146],data_range[178],data_range[210],data_range[241],data_range[285]], [1,45,78,108,147,179,211,242,286], save='no', normalized_type='no')
                print('Took %s minutes' % ((time.perf_counter()-start)/60))

client.close()
cluster.close()
print(datetime.datetime.now().isoformat())