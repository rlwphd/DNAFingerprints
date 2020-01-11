# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 08:23:45 2018

@author: Administrator
"""


import os
import math
import time
import datetime
import dask.dataframe as ddf
import dask.array as da
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier, RandomForestClassifier
from sklearn.model_selection import ShuffleSplit
from sklearn.linear_model import SGDClassifier, PassiveAggressiveClassifier, Perceptron
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from dask.distributed import Client, LocalCluster
from dask_ml.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from collections import Counter

''' You will need to change these folder locations to be the folder where you
    want to save the DNA sequences, the files of the created BOC reads, and
    the files of the created predicition arrays and confusion matrices. '''
# Estabilishing where things are being run and where things are being saved
bacteria = 'C:/Usr/Bacteria/'
local_BOC = 'C:/Usr/BOC/'
pred_arrays = 'C:/Usr/NPY/'


def pca_plot(data,name,dna_type,err,read):

    ymax_data = 1.1*max(data[:,1])
    ymin_data = 1.1*min(data[:,1])
    xmax_data = 1.1*max(data[:,0])
    xmin_data = 1.1*min(data[:,0])

    fig = plt.figure(figsize=(20,10))
    ax = plt.subplot()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    ax.set_ylabel('Principal Component 2', fontsize=22)
    ax.set_xlabel('Principal Component 1', fontsize=22)

    ax.set_ylim(ymin_data, ymax_data)
    ax.set_xlim(xmin_data, xmax_data)


    if dna_type == 'G':
        cb = [int_p for int_p,ii in enumerate(name[:10000]) if ii == '0']
        cc = [int_p for int_p,ii in enumerate(name[:10000]) if ii == '1']
        ceh = [int_p for int_p,ii in enumerate(name[:10000]) if ii == '8']
        cec = [int_p for int_p,ii in enumerate(name[:10000]) if ii == '2']
        cef = [int_p for int_p,ii in enumerate(name[:10000]) if ii == '9']
        ck = [int_p for int_p,ii in enumerate(name[:10000]) if ii == '3']
        cse = [int_p for int_p,ii in enumerate(name[:10000]) if ii == '4']
        csa = [int_p for int_p,ii in enumerate(name[:10000]) if ii == '5']
        cspn = [int_p for int_p,ii in enumerate(name[:10000]) if ii == '6']
        cspy = [int_p for int_p,ii in enumerate(name[:10000]) if ii == '7']
        ax.scatter(data[cb,0],data[cb,1], s=200, marker='o', c='xkcd:lime green', label='Bacteroides fragilis')
        ax.scatter(data[cc,0],data[cc,1], s=200, marker='o', c='xkcd:sienna', label='Campylobacter jejuni')
        ax.scatter(data[ceh,0],data[ceh,1], s=200, marker='o', c='xkcd:blue', label='Enterococcus hirae')
        ax.scatter(data[cec,0],data[cec,1], s=200, marker='o', c='xkcd:orange', label='Escherichia coli')
        ax.scatter(data[cef,0],data[cef,1], s=200, marker='o', c='xkcd:yellow', label='Escherichia fergusonii')
        ax.scatter(data[ck,0],data[ck,1], s=200, marker='o', c='xkcd:pink', label='Klebsiella pneumoniae')
        ax.scatter(data[cse,0],data[cse,1], s=200, marker='o', c='xkcd:red', label='Salmonella enterica')
        ax.scatter(data[csa,0],data[csa,1], s=200, marker='o', c='xkcd:violet', label='Staphylococcus aureus')
        ax.scatter(data[cspn,0],data[cspn,1], s=200, marker='o', c='xkcd:green', label='Streptococcus pneumoniae')
        ax.scatter(data[cspy,0],data[cspy,1], s=200, marker='o', c='xkcd:cyan', label='Streptococcus pyogenes')

        cb = [int(int_p+10000) for int_p,ii in enumerate(name[10000:20000]) if ii == '0']
        cc = [int(int_p+10000) for int_p,ii in enumerate(name[10000:20000]) if ii == '1']
        ceh = [int(int_p+10000) for int_p,ii in enumerate(name[10000:20000]) if ii == '8']
        cec = [int(int_p+10000) for int_p,ii in enumerate(name[10000:20000]) if ii == '2']
        cef = [int(int_p+10000) for int_p,ii in enumerate(name[10000:20000]) if ii == '9']
        ck = [int(int_p+10000) for int_p,ii in enumerate(name[10000:20000]) if ii == '3']
        cse = [int(int_p+10000) for int_p,ii in enumerate(name[10000:20000]) if ii == '4']
        csa = [int(int_p+10000) for int_p,ii in enumerate(name[10000:20000]) if ii == '5']
        cspn = [int(int_p+10000) for int_p,ii in enumerate(name[10000:20000]) if ii == '6']
        cspy = [int(int_p+10000) for int_p,ii in enumerate(name[10000:20000]) if ii == '7']
        ax.scatter(data[cb,0],data[cb,1], s=200, marker='^', c='xkcd:dark lime green')
        ax.scatter(data[cc,0],data[cc,1], s=200, marker='^', c='xkcd:brown')
        ax.scatter(data[ceh,0],data[ceh,1], s=200, marker='^', c='xkcd:darkblue')
        ax.scatter(data[cec,0],data[cec,1], s=200, marker='^', c='xkcd:orangered')
        ax.scatter(data[cef,0],data[cef,1], s=200, marker='^', c='xkcd:goldenrod')
        ax.scatter(data[ck,0],data[ck,1], s=200, marker='^', c='xkcd:magenta')
        ax.scatter(data[cse,0],data[cse,1], s=200, marker='^', c='xkcd:crimson')
        ax.scatter(data[csa,0],data[csa,1], s=200, marker='^', c='xkcd:plum')
        ax.scatter(data[cspn,0],data[cspn,1], s=200, marker='^', c='xkcd:darkgreen')
        ax.scatter(data[cspy,0],data[cspy,1], s=200, marker='^', c='xkcd:teal')

        ax.scatter(data[20000:21000,0],data[20000:21000,1], s=200, marker='^', label=name[20000], c='xkcd:salmon')
        ax.scatter(data[21000:22000,0],data[21000:22000,1], s=200, marker='^', label=name[21000], c='xkcd:grey')
        save_png = 'GPCA_%s_%s.png' %(str(err),str(read))
        plt.legend(loc='upper center',fontsize=16)

    elif dna_type == 'PI':
        
        ci = [int(int_p+250000) for int_p,ii in enumerate(name[250000:500000]) if ii == 0]
        ck = [int(int_p+250000) for int_p,ii in enumerate(name[250000:500000]) if ii == 1]
        cn = [int(int_p+250000) for int_p,ii in enumerate(name[250000:500000]) if ii == 2]
        cnr = [int(int_p+250000) for int_p,ii in enumerate(name[250000:500000]) if ii == 3]
        cv = [int(int_p+250000) for int_p,ii in enumerate(name[250000:500000]) if ii == 4]
        ax.scatter(data[ci,0], data[ci,1], s=50, marker='s', c='xkcd:crimson')
        ax.scatter(data[ck,0], data[ck,1], s=50, marker='s', c='xkcd:orangered')
        ax.scatter(data[cn,0], data[cn,1], s=50, marker='s', c='xkcd:darkgreen')
        ax.scatter(data[cv,0], data[cv,1], s=50, marker='s', c='xkcd:darkblue')
        ax.scatter(data[cnr,0], data[cnr,1], s=60, marker='D', c='xkcd:grape')

        ci = [int_p for int_p,ii in enumerate(name[:250000]) if ii == 0]
        ck = [int_p for int_p,ii in enumerate(name[:250000]) if ii == 1]
        cn = [int_p for int_p,ii in enumerate(name[:250000]) if ii == 2]
        cnr = [int_p for int_p,ii in enumerate(name[:250000]) if ii == 3]
        cv = [int_p for int_p,ii in enumerate(name[:250000]) if ii == 4]
        i = ax.scatter(data[ci,0], data[ci,1], s=60, marker='o', c='xkcd:red')
        k = ax.scatter(data[ck,0], data[ck,1], s=60, marker='o', c='xkcd:orange')
        n = ax.scatter(data[cn,0], data[cn,1], s=60, marker='o', c='xkcd:green')
        v = ax.scatter(data[cv,0], data[cv,1], s=60, marker='o', c='xkcd:blue')
        nr = ax.scatter(data[cnr,0], data[cnr,1], s=50, marker='^', c='xkcd:violet')

        save_png = 'P_Ind_PCA_%s_%s.png' %(str(err),str(read))
        plt.legend([i,k,n,v,nr],['IMP','KPC', 'NDM', 'VIM', 'No Resistance'],loc='upper center',fontsize=16)

    elif dna_type == 'PG':
        cnr = [int(int_p+250000) for int_p,ii in enumerate(name[250000:500000]) if ii == 0]
        cr = [int(int_p+250000) for int_p,ii in enumerate(name[250000:500000]) if ii == 1]
        ax.scatter(data[cr,0], data[cr,1], s=50, marker='s', c='xkcd:crimson')
        ax.scatter(data[cnr,0], data[cnr,1], s=60, marker='D', c='xkcd:darkgreen')

        cnr = [int_p for int_p,ii in enumerate(name[:250000]) if ii == 0]
        cr = [int_p for int_p,ii in enumerate(name[:250000]) if ii == 1]
        r = ax.scatter(data[cr,0], data[cr,1], s=60, marker='o', c='xkcd:red')
        nr = ax.scatter(data[cnr,0], data[cnr,1], s=50, marker='^', c='xkcd:green')

        save_png = 'P_Group_PCA_%s_%s.png' %(str(err),str(read))
        plt.legend([r,nr],['Resistance','No Resistance'],loc='upper center',fontsize=16)

    else:
        return

    plt.rc('xtick',labelsize=20)
    plt.rc('ytick',labelsize=20)
    plt.savefig(save_png, dpi='figure', bbox_inches='tight')
    plt.close()
    return


def str_count(str_part):
    A = str_part.count('A')
    T = str_part.count('T')
    G = str_part.count('G')
    C = str_part.count('C')

    return (A, T, G, C), (T, A, C, G)

def kmer_fingerprints(whole_str,dna_length,kmer_range):
    str_part = (whole_str[ii:ii+dna_length] for ii in range(len(whole_str)-dna_length+1))
    kmer_list = [item for string in str_part for item in str_count(string)]
    kmer_dict = Counter(kmer_list)
    results = [kmer_dict[val] for val in kmer_range]

    return results

def title_extraction(header, resistance):
    heading = header.split(' ')
    seq_record_id = heading[0]
    genus = heading[1]
    if len(heading) >= 3:
        species = heading[2]
        name = genus + ' ' + species
    else:
        name = genus
        
    for word in heading:
        if 'plasmid' in word:
            dna_type = 'Plasmid'
            break
        elif 'genome' in word:
            dna_type = 'Genome'
            break
        elif 'sequence'in word:
            dna_type = 'Sequence'
            break
        else:
            dna_type = ""
    
    bacteria_type = ""
    strain = ""
    data_add = [seq_record_id[1:], resistance, name, genus, dna_type, strain, bacteria_type,  header]
    
    return data_add

def multiple_str_check(file):
    header = []
    start = []
    with open(file, 'r') as f:
        for ii, line in enumerate(f):
            if '>' == line[0]:
                header.append(line)
                start.extend([ii+1])
            end = ii+1
        start.extend([end])
    return header, start

def str_extraction(file,start,end,dna_length):
    string = []
    with open(file,'r') as f:
        for ii, line in enumerate(f):
            if ii >= start and ii < end:
                string.append(line)
            elif ii == end:
                break
                
    
    section = ''.join(line.strip() for line in string)
    whole_string = ''.join([section,section[0:dna_length-1]])
    return whole_string

def kmer_main(file,dna_length,kmer_range,data_index,df_Genome,df_Plasmid, resistance):
    header,start = multiple_str_check(file)
    if len(header) > 1:
        for ii in range(len(header)):
            whole_str = str_extraction(file,start[ii],start[ii+1]-1,dna_length)
            species_data = title_extraction(header[ii], resistance)
            kmer_results = kmer_fingerprints(whole_str,dna_length,kmer_range)
            species_data.extend(kmer_results)
            df_kmer = pd.DataFrame(species_data, index=data_index)
            if species_data[4] != 'Plasmid':
                df_Genome = df_Genome.append(df_kmer.T, ignore_index=True)
            elif species_data[4] == 'Plasmid':
                df_Plasmid = df_Plasmid.append(df_kmer.T, ignore_index=True)
    else:
        whole_str = str_extraction(file,start[0],start[1]-1,dna_length)
        species_data = title_extraction(header[0], resistance)
        kmer_results = kmer_fingerprints(whole_str,dna_length,kmer_range)
        species_data.extend(kmer_results)
        df_kmer = pd.DataFrame(species_data, index=data_index)
        if species_data[4] != 'Plasmid':
            df_Genome = df_Genome.append(df_kmer.T, ignore_index=True)
        elif species_data[4] == 'Plasmid':
            df_Plasmid = df_Plasmid.append(df_kmer.T, ignore_index=True)
    
    type_change = {}
    for ii,name in enumerate(data_index):
        if ii < 8:
            type_change[name] = 'object'
        else:
            type_change[name]='int32'
    df_Genome = df_Genome.astype(type_change)
    df_Plasmid = df_Plasmid.astype(type_change)
    
    return df_Genome,df_Plasmid

def kmer_length():
    k = None
    while k is None:
        input_value = input("Please enter DNA segment length (5-100): ")
        try:
        # try and convert the string input to a number
            k = int(input_value)
            if k < 5:
                print("{input} is not a valid integer, please enter a valid integer between 5-100".format(input=input_value))
                k = None
            elif k > 100:
                print("{input} is not a valid integer, please enter a valid integer between 5-100".format(input=input_value))
                k = None
        except ValueError:
        # tell the user off
            print("{input} is not a valid integer, please enter a valid integer between 5-100".format(input=input_value))
    return k

def SERS_values(sers,categories,num_reads,arr,mutate,mutation):

    jj = 0
    while 1:
    # Randomizing the pyramid array
        mr = np.random.permutation(arr)
    # Adding in the mutations
        mr = np.where(mutate > 0, mutation, mr)
    # Getting the respective kmer counts and dividing the values 
    # by the total value to get the frequencies
        for nn in range(len(num_reads)):
        # Setting the limits for how much of the sequence it's using
            if num_reads[nn] > 10000:
                ff = 10000
                max_depth = int(num_reads[nn]/10000)
            else:
                ff = num_reads[nn]
                max_depth = 1 
            for mm in range(mr.shape[1]):
                for kk in range(max_depth):
                    sers[nn,mm,jj,:] += np.bincount(mr[kk,mm,:ff],minlength=categories)
                    
        jj+=1
        if jj >= sers.shape[2]:
            break
    sers /= np.array(num_reads).reshape((-1,1,1,1))
    return sers

def SERS_reads(dna_length,df,group,DNAtype,data_categories,bias,num_training_samples,num_reads,error_rate,SERS_location):

#Dividing the number of occurences of each bin by the total number of occurences
    df_prob = df.loc[:,data_categories[0]:data_categories[-1]].div(df.loc[:,data_categories[0]:data_categories[-1]].sum(axis=1),axis=0)

    for ii in range(len(df_prob.index)):
    # Getting just the probility values from the kmer counts
        prob = df_prob.iloc[ii,:].values
    # Getting the probility for the largest pyramid of interest and setting the data type
        read = np.random.RandomState(seed=231).choice(len(data_categories),(int(2*max(num_reads)/10000),10000),p=prob)
        read = np.stack([read for _ in range(len(error_rate))],axis=1)
        
    # Creating the mutation array
        mutate = np.zeros((int(2*max(num_reads)/10000),len(error_rate),10000),dtype=np.int16)
        for int_mut,mut in enumerate(error_rate):
            mutations = np.concatenate([np.random.RandomState(seed=123).choice([0,1],min(num_reads),p=[1-mut,mut]) for _ in range(int(2*max(num_reads)/min(num_reads)))]).reshape((-1,10000))
            mutate[:,int_mut,:] = mutations
        mutation = np.random.RandomState(seed=321).choice(len(data_categories),(int(2*max(num_reads)/10000),10000),p=bias)
        mutation = np.stack([mutation for _ in range(len(error_rate))],axis=1)
        
    # Getting the training samples for the species
        sers = np.empty((len(num_reads),len(error_rate),num_training_samples,len(data_categories)))
        sers_results = SERS_values(sers,len(data_categories),num_reads,read,mutate,mutation)
    # Subtracting off the random bias 
        sers_results -= bias
        
    # Cycling through each mutation array
        for mut_int in range(len(error_rate)):
        # Cycling through each pyramid size
            for read_int in range(len(num_reads)):
            # Storing the files as an hdf5 file
                with pd.HDFStore(os.path.join(local_BOC[SERS_location],'SERS_%s_%s_%s_%s_%smer_data.h5' % (str(int(error_rate[mut_int]*100)), str(num_reads[read_int]), group, DNAtype, str(dna_length))),complevel=9,complib='zlib') as store:
                # Putting it into a pandas dataframe and storing it
                    df_SERS = pd.DataFrame(sers_results[read_int,mut_int,:,:], columns=data_categories)
                    df_SERS['Name'] = [df.iloc[ii,1]]*len(df_SERS.index)
                    store.append('df%s' % (str(ii)), df_SERS, min_itemsize = {'values': 50})
        
        if ii % 10 == 0:
            print(ii)
            print('Saved %s' % (datetime.datetime.now().isoformat()))
    return


#dna_length=kmer_length()
dna_length = 10
# Recording the time it takes to run everything
begin = datetime.datetime.now().isoformat()
start = time.perf_counter()
# number of samples per species 
num_training_samples = 1000
# error rate
error_rate = [0,0.01,0.05,0.1,0.25,0.33,0.5,0.75,0.9,1]
# number of optical sequencing reads
num_reads = [100,1000,10000,100000,1000000]

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

# getting the list of all of the folders and files that have the DNA sequences in them
file_list = [(os.path.join(root,name),root[3:].split('/')[-1],name) for root, dirs, files in os.walk(bacteria) for name in files if name.endswith(".txt") or name.endswith(".fna")]

# Running through all of the DNA sequence files
for int_file,file in enumerate(file_list):
# creating the empty dataframes to store the data in
    df_Genome = pd.DataFrame(columns=data_index)
    df_Plasmid = pd.DataFrame(columns=data_index)

# creating the 10mer data files for the DNA sequences
    resistance = file[2][:-4]
    df_Genome, df_Plasmid = kmer_main(file[0],dna_length,kmer_range, data_index,df_Genome,df_Plasmid, resistance)
    print('Completed %s' % file[2][:-4])

# saving the dataframe
    if len(df_Genome.index) > 0:
        df_Genome.to_hdf('PandasDataFrame_%s_Genome_%smer_data.h5' % (file[1],str(dna_length)),'df%s' % (int_file),mode='a',format='table')
    if len(df_Plasmid.index) > 0:
        df_Plasmid.to_hdf('PandasDataFrame_%s_Plasmid_%smer_data.h5' % (file[1],str(dna_length)),'df%s' % (int_file),mode='a',format='table')
    print('file saved')

    del(df_Genome)
    del(df_Plasmid)

# Running the code for creating the simulated SERS data
for file in os.listdir(os.getcwd()):    
    """ Splitting file name up into parts to determine if its genomic dna and to get the dna length """
    file_split = file[:-3].split('_')
    if file_split[0] == 'PandasDataFrame' :
    # get dna length from file name
#        dna_length = int(file_split[3].replace('mer',''))
    # collecting the dataframe from the stored file for analysis
        df = ddf.read_hdf(os.path.join(os.getcwd(),file), 'df*').compute()
        SERS_reads(dna_length,df,file_split[1],file_split[2],data_categories,bias,num_training_samples,num_reads,error_rate,local_BOC)

end = time.perf_counter()
print('# of hours to run code: %s' % ((end-start)/3600))
print(datetime.datetime.now().isoformat())

# Getting the list of all the data files that need to be tested
file_list = [file for dirr in local_BOC for file in os.listdir(dirr)]
# Getting the list of the files for the specific reads and errors
working_genome_list = [file for error in error_rate for reads in num_reads for file in file_list if '_%s_%s_' % (int(100*error),reads) in file and 'Genome' in file]
working_plasmid_list = [file for error in error_rate for reads in num_reads for file in file_list if '_%s_%s_' % (int(100*error),reads) in file and 'Plasmid' in file]

# Getting each file in the list of files to use
train_file = []
test_file = []
extra_file = []
for file in working_genome_list:
    if file.split('_')[3] == 'Training':
        train_file.append(file)
    elif file.split('_')[3] == 'Testing':
        test_file.append(file)
    elif file.split('_')[3] == 'Extras':
        extra_file.append(file)

# Number of testing plasmids
num_plasmid = 100
group_plasmid = 400
# Number of training bacterial species
bac_type = 10
# Number of training plasmid types
plasmid_type = 5
# Number of SERS readings to grab at a time from the training samples
num_SERS = 50
# Number of cross fold validation
cfv = 10
# Number of samples to train on
samples = 250000

# Determining how many times you want to test the ML algorithm
# against different unseen sets of plasmids/genomes
unseen_tests = 2
unseen_split = ShuffleSplit(unseen_tests,None,num_plasmid,123)
# Split for looking at resistance vs non-resistance grouping for non-resitant plasmids
unseen_group_split = ShuffleSplit(unseen_tests,None,group_plasmid,123)

# Creating the list of all of the classifiers for storing things
classifier_list = ['SGD','Perceptron','Passive-Aggressive','Neural Network','GNB','BNB','RF','ET','GB','PCA_svd','LDA','QDA']
g_class = ['Bacteroides fragilis','Campylobacter jejuni','Escherichia coli','Klebsiella pneumoniae','Salmonella enterica','Staphylococcus aureus','Streptococcus pneumoniae','Streptococcus pyogenes','Enterococcus hirae','Escherichia fergusonii','Klebsiella aerogenes','Mycobacterium tuberculosis']
g_abr_class = ['B. fragilis','C. jejuni','E. coli','K. pneumoniae','S. enterica','S. aureus','S. pneumoniae','S. pyogenes','E. hirae','E. fergusonii','K. aerogenes','M. tuberculosis']
p_class = ['IMP','KPC','NDM','No Resistance','VIM']
pg_class = ['No Resistance','Resistance']

# Intializing dask to run things in parallel
with LocalCluster(processes=False) as cluster, Client(cluster) as client:
    
# Retrieving wether the file is genomic or plasmid
    file_type = 'Genome'
# Opening up each of the HDF5 files in the working list
    store_train = ddf.read_hdf([os.path.join(local_BOC,file) for file in train_file], 'df*')
    store_test = ddf.read_hdf([os.path.join(local_BOC,file) for file in test_file], 'df*')
    store_extra = ddf.read_hdf([os.path.join(local_BOC,file) for file in extra_file], 'df*')
# Splitting the files into the unseen sets and the training sets
    train_set = store_train.drop('Name',axis='columns').to_dask_array(True)
    unseen_test_set = store_test.drop('Name',axis='columns').to_dask_array(True)
    check_set = store_extra.drop('Name',axis='columns').to_dask_array(True)
    bac_train = np.char.array(store_train.loc[:,'Name'].compute().to_list())
    bac_test = np.char.array(store_test.loc[:,'Name'].compute().to_list())
    bac_ex = np.char.array(store_extra.loc[:,'Name'].compute().to_list())
        
# Converting the names into numbers for the analysis
    bac_train = np.where(bac_train.rfind('_') != bac_train.find('_'), bac_train.rpartition('_')[:,0].replace('_',' '),bac_train.replace('_',' '))
    bac_train = np.where(bac_train == 'Bacteriodes fragilis', g_class[0], bac_train)
    bac_train = np.where(bac_train == 'Streptococcus pneumonia', g_class[6], bac_train)
    bac_test = np.where(bac_test.rfind('_') != bac_test.find('_'), bac_test.rpartition('_')[:,0].replace('_',' '),bac_test.replace('_',' '))
    bac_ex = np.where(bac_ex.rfind('_') != bac_ex.find('_'), bac_ex.rpartition('_')[:,0].replace('_',' '),bac_ex.replace('_',' '))
    bac_ex = np.where(bac_ex == 'Enterobacter aerogenes', g_class[10], bac_ex)
    
    bac_train = np.where(bac_train == g_class[0], 0, bac_train)
    bac_train = np.where(bac_train == g_class[1], 1, bac_train)
    bac_train = np.where(bac_train == g_class[2], 2, bac_train)
    bac_train = np.where(bac_train == g_class[3], 3, bac_train)
    bac_train = np.where(bac_train == g_class[4], 4, bac_train)
    bac_train = np.where(bac_train == g_class[5], 5, bac_train)
    bac_train = np.where(bac_train == g_class[6], 6, bac_train)
    bac_train = np.where(bac_train == g_class[7], 7, bac_train)
    bac_train = np.where(bac_train == g_class[8], 8, bac_train)
    bac_train = np.where(bac_train == g_class[9], 9, bac_train)
    bac_train = bac_train.astype('int')
    
    bac_test = np.where(bac_test == g_class[0], 0, bac_test)
    bac_test = np.where(bac_test == g_class[1], 1, bac_test)
    bac_test = np.where(bac_test == g_class[2], 2, bac_test)
    bac_test = np.where(bac_test == g_class[3], 3, bac_test)
    bac_test = np.where(bac_test == g_class[4], 4, bac_test)
    bac_test = np.where(bac_test == g_class[5], 5, bac_test)
    bac_test = np.where(bac_test == g_class[6], 6, bac_test)
    bac_test = np.where(bac_test == g_class[7], 7, bac_test)
    bac_test = np.where(bac_test == g_class[8], 8, bac_test)
    bac_test = np.where(bac_test == g_class[9], 9, bac_test)
    bac_test = bac_test.astype('int')
    
# Creating a testing data array for the unseen tests and the training data
    train_data = da.concatenate([train_set[int(num_SERS*ii+jj*num_training_samples+kk*bac_type*num_training_samples):int(num_SERS*(ii+1)+jj*num_training_samples+kk*bac_type*num_training_samples),:] for kk in range(int(len(error_rate)*len(num_reads))) for ii in range(int(num_training_samples/num_SERS)) for jj in range(bac_type)],axis=0).rechunk({0: 500})
    unseen_data = da.concatenate([unseen_test_set[int(num_SERS*ii+jj*num_training_samples+kk*bac_type*num_training_samples):int(num_SERS*(ii+1)+jj*num_training_samples+kk*bac_type*num_training_samples),:] for kk in range(int(len(error_rate)*len(num_reads))) for ii in range(int(num_training_samples/num_SERS)) for jj in range(bac_type)],axis=0).rechunk({0: 500})
# Creating the labels/categories for the data
    y_train_data = da.concatenate([bac_train[int(num_SERS*ii+jj*num_training_samples+kk*bac_type*num_training_samples):int(num_SERS*(ii+1)+jj*num_training_samples+kk*bac_type*num_training_samples)] for kk in range(int(len(error_rate)*len(num_reads))) for ii in range(int(num_training_samples/num_SERS)) for jj in range(bac_type)],axis=0).rechunk({0: 500})
    y_unseen_data = da.concatenate([bac_test[int(num_SERS*ii+jj*num_training_samples+kk*bac_type*num_training_samples):int(num_SERS*(ii+1)+jj*num_training_samples+kk*bac_type*num_training_samples)] for kk in range(int(len(error_rate)*len(num_reads))) for ii in range(int(num_training_samples/num_SERS)) for jj in range(bac_type)],axis=0).rechunk({0: 500})
            
    print('Moving on to classifying')
# Cycling through the 24 combinations of error rate and number of reads
    for int_err, error in enumerate(error_rate):
    # Recording the time it takes to run everything
        start1 = time.perf_counter()
        for int_read, reads in enumerate(num_reads):
        # Creating a pandas dataframe for the results
            results = pd.DataFrame(index = np.arange(cfv))
        # Getting the values for the unseen testing
            xu, yu = unseen_data[int(bac_type*num_training_samples*(int_read+len(num_reads)*int_err)):int(bac_type*num_training_samples*(1+int_read+len(num_reads)*int_err))].compute(), y_unseen_data[int(bac_type*num_training_samples*(int_read+len(num_reads)*int_err)):int(bac_type*num_training_samples*(1+int_read+len(num_reads)*int_err))].compute()
        # Getting the values for the extras testing
            xe, ye = check_set[int(2*num_training_samples*(int_read+len(num_reads)*int_err)):int(2*num_training_samples*(1+int_read+len(num_reads)*int_err))].compute(), np.tile(np.array([0,1,2,3,4,5,6,7,8,9]), int(np.ceil(2*num_training_samples/9)))[:int(2*num_training_samples)]

        # Creating the split for the 90% train, 10% validate split for the training set
            for cv in range(cfv):
            # Here are the classifiers that support the `partial_fit` method
                partial_fit_classifiers = {
                    'SGD': SGDClassifier(max_iter=1000,tol=1e-3,random_state=123),
                    'Perceptron': Perceptron(max_iter=1000,tol=1e-3,random_state=123),
                    'Passive-Aggressive': PassiveAggressiveClassifier(max_iter=1000,tol=1e-3,random_state=123),
                    'Neural Network': MLPClassifier(max_iter=1000,random_state=123),
                    'GNB': GaussianNB(),
                    'BNB': BernoulliNB(),
                    'RF': RandomForestClassifier(n_estimators = 10,max_features=None,bootstrap=False,random_state=123,warm_start=True),
                    'ET': ExtraTreesClassifier(n_estimators = 10,max_features=None,bootstrap=False,random_state=123,warm_start=True),
                    'GB': GradientBoostingClassifier(n_estimators = 10,max_features=None,random_state=123,warm_start=True),
                    'LDA': LinearDiscriminantAnalysis(solver='svd'),
                    'QDA': QuadraticDiscriminantAnalysis()
                }
            # Timing how long it takes to run through all models
                tick = time.perf_counter()
            # Getting the split for this cv
                x_train, x_test, y_train, y_test = train_test_split(train_data[int(bac_type*num_training_samples*(int_read+len(num_reads)*int_err)):int(bac_type*num_training_samples*(1+int_read+len(num_reads)*int_err))], y_train_data[int(bac_type*num_training_samples*(int_read+len(num_reads)*int_err)):int(bac_type*num_training_samples*(1+int_read+len(num_reads)*int_err))], random_state=123+cv)
            # Getting the values for the model for training
                x, y = x_train.compute(), y_train.compute()
            # Getting the values for the model for testing
                xt, yt = x_test.compute(), y_test.compute()
                np.save(os.path.join(pred_arrays, '%s %s Genome Train-test values %s.npy' % (int(error*100), reads, cv)), yt)
                np.save(os.path.join(pred_arrays, '%s %s Genome Unseen values %s.npy' % (int(error*100), reads, cv)), yu)
            # Iterating through the different models
                for cls_name, cls in partial_fit_classifiers.items():
                # timing each classifier
                    tick1 = time.perf_counter()
                # Using the client to parallelize the fits
                    with joblib.parallel_backend('dask'):
                    # update estimator with data
                        cls.fit(x, y)
                    # getting how long it takes to fit the data
                        fit_time = time.perf_counter() - tick1
                    # scoring the testing set
                        acc = cls.score(xt, yt)
                    # getting the prediction arrays
                        if cls_name == 'SGD' or cls_name == 'Perceptron' or cls_name == 'Passive-Aggressive':
                            pred = cls.decision_function(xt)
                        else:
                            pred = cls.predict_proba(xt)
                # Recording results
                    if cv == 0:
                    # model fitting stats
                        results.loc[cv,'%s %s %s Genome Train Time' % (error, reads, cls_name)] = np.mean(fit_time)
                        results.loc[cv,'%s %s %s Genome Train Sample Number' % (error, reads, cls_name)] = len(y)
                # accumulate test accuracy stats
                    results.loc[cv,'%s %s %s Genome Train Accuracy' % (error, reads, cls_name)] = np.mean(acc)
                # Calculating Confusion Matrix
                    conf_mat = confusion_matrix(yt, pred.argmax(axis=1))
                    np.save(os.path.join(pred_arrays, '%s %s %s Genome Prediction Array Train %s.npy' % (int(error*100), reads, cls_name, cv)), pred)
                    np.save(os.path.join(pred_arrays, '%s %s %s Genome Confusion Matrix Train %s.npy' % (int(error*100), reads, cls_name, cv)), conf_mat)
                # Printing the time the model testing has been running
                    print('Train set tested. Total time model testing has been running %s minutes' % ((time.perf_counter()-tick)/60))
                # timing each classifier
                    tick1 = time.perf_counter()
                # Using the client to parallelize the fits
                    with joblib.parallel_backend('dask'):
                    # update estimator with data
                        acc = cls.score(xu, yu)
                        if cls_name == 'SGD' or cls_name == 'Perceptron' or cls_name == 'Passive-Aggressive':
                            pred = cls.decision_function(xu)
                            check = cls.decision_function(xe)
                        else:
                            pred = cls.predict_proba(xu)
                            check = cls.predict_proba(xe)
                    test_time = time.perf_counter() - tick1
                    if cv == 0:
                    # model fitting stats
                        results.loc[cv,'%s %s %s Genome Test Time' % (error, reads, cls_name)] = np.mean(test_time)
                        results.loc[cv,'%s %s %s Genome Test Number' % (error, reads, cls_name)] = len(yu)
                # accumulate test accuracy stats
                    results.loc[cv,'%s %s %s Genome Test Accuracy' % (error, reads, cls_name)] = np.mean(acc)
                # Calculating Confusion Matrix
                    conf_mat = confusion_matrix(yu, pred.argmax(axis=1))
                    np.save(os.path.join(pred_arrays, '%s %s %s Genome Prediction Array Test %s.npy' % (int(error*100), reads, cls_name, cv)), pred)
                    np.save(os.path.join(pred_arrays, '%s %s %s Genome Confusion Matrix Test %s.npy' % (int(error*100), reads, cls_name, cv)), conf_mat)
                    conf_mat = confusion_matrix(ye, check.argmax(axis=1))
                    np.save(os.path.join(pred_arrays, '%s %s %s Genome Extras Prediction Array Test %s.npy' % (int(100*error), reads, cls_name, cv)), check)
                    np.save(os.path.join(pred_arrays, '%s %s %s Genome Extras Confusion Matrix Test %s.npy' % (int(error*100), reads, cls_name, cv)), conf_mat)
                print('Took %s minutes to fit, validate and test all models for 1 cross-fold validation' % ((time.perf_counter()-tick)/60))
                print('%s fold Cross Validation finished' % (cv))

                if cv == 0:
                # Timing how long it takes to run through all discriminant analysis models
                    tick = time.perf_counter()
                ## Creating component analysis data
                    cls = PCA(random_state=123)
                    g_pca_train = np.array(cls.fit_transform(np.concatenate((x,xt),axis=0)))
                    g_pca_unseen = np.array(cls.transform(xu))
                    g_pca_check = np.array(cls.transform(xe))
                    g_pca = np.concatenate((g_pca_train,g_pca_unseen,g_pca_check),axis=0)
                    np.save(os.path.join(pred_arrays, '%s %s Genome PCA.npy' % (int(100*error), reads)), g_pca)
                    
                    y_name = np.concatenate((y,yt,yu),axis=0)
                    g_name = np.concatenate((y_name,bac_ex[int(2*num_training_samples*(int_read+len(num_reads)*int_err)):int(2*num_training_samples*(1+int_read+len(num_reads)*int_err))]),axis=0)
                    np.save(os.path.join(pred_arrays, '%s %s Genome PCA Names.npy' % (int(100*error), reads)), g_name)
                    
                    pca_plot(g_pca[:,:2],g_name,'G',int(100*error),reads)

        #Saving the results
            results.to_csv('%s_ML_Results_%s_%s.csv' % (file_type,reads,int(100*error)))
            print('Results saved')
                       
            print('%s error rate and %s number of reads tests finished' % (error, reads))
        print('Took %s minutes to run all number of reads tests' % ((time.perf_counter()-start1)/60))
    print('Genome Finished. Taken %s hours to run code.' % ((time.perf_counter()-start)/(3600)))
    print(datetime.datetime.now().isoformat())

# Creating a dictionary for the results
    pred = {}
    for cls_name in classifier_list:
        pred[cls_name] = []    
# Testing the machine learning techniques on different sets of data being the never before seen test set
    for int_data in range(unseen_tests):
    # Retrieving wether the file is genomic or plasmid
        file_type = 'Plasmid'
    # Determining how long this takes
        tick = time.perf_counter()
    # Splitting the files into the unseen sets and the training sets
        train_set = []
        unseen_test_set = []
        train_group_set = []
        unseen_group_set = []
        file_info = []
    # Getting each file in the list of files to use
        for int_list,file in enumerate(working_plasmid_list):
        # Opening up each of the HDF5 files in the working list
            store = ddf.read_hdf(os.path.join(local_BOC,file), 'df*')
        # Getting the file type and the number of plasmids/genomes in the file
            file_info.append((file.split('_')[3],store.npartitions))
        # Creating a split of 100 training samples and the rest for testing as never before seen
            split = unseen_split.split(np.zeros((file_info[int_list][1],1)))
        # Grabbing 400 training samples for Group testing for the NonResistant sample 
            if file_info[int_list][0] == 'NonResistant':
                split_group = unseen_group_split.split(np.zeros((file_info[int_list][1],1)))
        # Getting the split of data for this learning test with never before seen sets
            for int_next in range(int_data+1):
            # Getting the correct data split
                train,unseen_ind = next(split)
                if file_info[int_list][0] == 'NonResistant': 
                    train_group,unseen_group = next(split_group)
        # Storing just the values of the data split
            train_set.append(store.drop('Name',axis='columns').partitions[train].to_dask_array(True))
            unseen_test_set.append(store.drop('Name',axis='columns').partitions[unseen_ind].to_dask_array(True))
            if file_info[int_list][0] == 'NonResistant':
                train_group_set.append(store.drop('Name',axis='columns').partitions[train_group].to_dask_array(True))
                unseen_group_set.append(store.drop('Name',axis='columns').partitions[unseen_group].to_dask_array(True))
            else:
                train_group_set.append(store.drop('Name',axis='columns').partitions[train].to_dask_array(True))
                unseen_group_set.append(store.drop('Name',axis='columns').partitions[unseen_ind].to_dask_array(True))
        print('Took %s hours to extract data' % ((time.perf_counter()-tick)/3600))
    # Removing unwanted variables
        del(train)
        del(unseen_ind)
        del(split)
        del(train_group)
        del(unseen_group)
        del(split_group)
        del(int_list)
        del(int_next)

    # Getting what set of files to grab
        int_set = 0
    # Cycling through the 24 combinations of error rate and number of reads
        for error in error_rate:
        # Recording the time it takes to run everything
            start1 = time.perf_counter()
            for reads in num_reads:
            # Creating a pandas dataframe for the results
                results = pd.DataFrame(index = np.arange(cfv))
                results_group = pd.DataFrame(index = np.arange(cfv))
            #Creating lists to save the dictionaries into for cross fold validation
                cv_stats = []
            
            # Creating a testing data array for the unseen tests and the training data
                train_data = da.concatenate([arr[int(num_SERS*ii+jj*num_training_samples):int(num_SERS*(ii+1)+jj*num_training_samples)] for ii in range(int(num_training_samples/num_SERS)) for arr in train_set[int(int_set*plasmid_type):int((int_set+1)*plasmid_type)] for jj in range(len(arr.chunks[0]))],axis=0).rechunk({0: 25000})
                unseen_data = da.concatenate([arr[int(num_SERS*ii+jj*num_training_samples):int(num_SERS*(ii+1)+jj*num_training_samples)] for ii in range(int(num_training_samples/num_SERS)) for arr in unseen_test_set[int(int_set*plasmid_type):int((int_set+1)*plasmid_type)] for jj in range(len(arr.chunks[0]))],axis=0).rechunk({0: 25000})
                train_group_data = da.concatenate([arr[int(num_SERS*ii+jj*num_training_samples):int(num_SERS*(ii+1)+jj*num_training_samples)] for ii in range(int(num_training_samples/num_SERS)) for arr in train_group_set[int(int_set*plasmid_type):int((int_set+1)*plasmid_type)] for jj in range(len(arr.chunks[0]))],axis=0).rechunk({0: 25000})
                unseen_group_data = da.concatenate([arr[int(num_SERS*ii+jj*num_training_samples):int(num_SERS*(ii+1)+jj*num_training_samples)] for ii in range(int(num_training_samples/num_SERS)) for arr in unseen_group_set[int(int_set*plasmid_type):int((int_set+1)*plasmid_type)] for jj in range(len(arr.chunks[0]))],axis=0).rechunk({0: 25000})
            # Creating the labels/categories for the data
                y_train_data = da.concatenate([np.ones(int(num_SERS))*int_arr for ii in range(int(num_training_samples/num_SERS)) for int_arr,arr in enumerate(train_set[int(int_set*plasmid_type):int((int_set+1)*plasmid_type)]) for jj in range(len(arr.chunks[0]))],axis=0).rechunk({0: 25000})
                y_unseen_data = da.concatenate([np.ones(int(num_SERS))*int_arr for ii in range(int(num_training_samples/num_SERS)) for int_arr,arr in enumerate(unseen_test_set[int(int_set*plasmid_type):int((int_set+1)*plasmid_type)]) for jj in range(len(arr.chunks[0]))],axis=0).rechunk({0: 25000})
                y_train_group = da.concatenate([np.ones(int(num_SERS)) if file_info[int_arr][0] != 'NonResistant' else np.zeros(int(num_SERS)) for ii in range(int(num_training_samples/num_SERS)) for int_arr,arr in enumerate(train_group_set[int(int_set*plasmid_type):int((int_set+1)*plasmid_type)]) for jj in range(len(arr.chunks[0]))],axis=0).rechunk({0: 25000})
                y_unseen_group = da.concatenate([np.ones(int(num_SERS)) if file_info[int_arr][0] != 'NonResistant' else np.zeros(int(num_SERS)) for ii in range(int(num_training_samples/num_SERS)) for int_arr,arr in enumerate(unseen_group_set[int(int_set*plasmid_type):int((int_set+1)*plasmid_type)]) for jj in range(len(arr.chunks[0]))],axis=0).rechunk({0: 25000})
            # Upping the set of files to grab
                int_set += 1
            
            # Getting the # of times the model needs to be iterated on
                if len(unseen_data) % samples != 0:
                    model_unseen_count = int(len(unseen_data)/samples) + 1
                else:
                    model_unseen_count = int(len(unseen_data)/samples)
                
            # Creating the split for the 90% train, 10% validate split for the training set
                for cv in range(cfv):
                # Here are the classifiers that support the `partial_fit` method
                    partial_fit_classifiers = {
                        'SGD': SGDClassifier(random_state=123),
                        'Perceptron': Perceptron(tol=1e-3,random_state=123),
                        'Passive-Aggressive': PassiveAggressiveClassifier(tol=1e-3,random_state=123),
                        'Neural Network': MLPClassifier(random_state=123),
                        'GNB': GaussianNB(),
                        'BNB': BernoulliNB(),
                        'RF': RandomForestClassifier(n_estimators=10,max_features=None,bootstrap=False,random_state=123,warm_start=True),
                        'ET': ExtraTreesClassifier(n_estimators=10,max_features=None,bootstrap=False,random_state=123,warm_start=True),
                        'GB': GradientBoostingClassifier(n_estimators=10,max_features=None,random_state=123,warm_start=True)
                    }
                # Timing how long it takes to run through all models
                    tick = time.perf_counter()
                # Getting the split for this cv
                    x_train, x_test, y_train, y_test = train_test_split(train_data, y_train_data, random_state=123+cv)
                # Getting the # of times the model needs to be iterated on
                    if len(x_train) % samples != 0:
                        model_fit_count = int(len(x_train)/samples) + 1
                    else:
                        model_fit_count = int(len(x_train)/samples)
                    if len(x_test) % samples != 0:
                        model_test_count = int(len(x_test)/samples) + 1
                    else:
                        model_test_count = int(len(x_test)/samples)
                # Iterating through the # of fits
                    for int_model in range(model_fit_count): 
                    # Getting the values for the model for testing
                        if int_model != model_fit_count-1:
                            x, y = x_train[samples*int_model:samples*(int_model+1)].compute(), y_train[samples*int_model:samples*(int_model+1)].compute()
                        else:
                            x, y = x_train[samples*int_model:].compute(), y_train[samples*int_model:].compute()

                    # Creating the timing variable
                        fit_time = [np.zeros(model_fit_count) for ii in partial_fit_classifiers]
                    # Iterating through the different models
                        for int_cls, (cls_name, cls) in enumerate(partial_fit_classifiers.items()):
                        # timing each classifier
                            tick1 = time.perf_counter()
                        # Using the client to parallelize the fits
                            with joblib.parallel_backend('dask'):
                            # update estimator with data
                                if cls_name == 'RF' or cls_name == 'ET' or cls_name == 'GB':
                                    cls.fit(x, y)
                                    if int_model != model_fit_count-1:
                                        cls.n_estimators += 1
                                else:
                                    cls.partial_fit(x, y, classes=np.arange(plasmid_type))
                            fit_time[int_cls][int_model] = time.perf_counter() - tick1
                # Printing the time it took to fit all of the models
                    print('Took %s minutes to fit all models' % ((time.perf_counter()-tick)/60))
                # Creating the test variables
                    acc = [np.zeros(model_test_count) for ii in range(len(partial_fit_classifiers))]
                # Iterating through the # of tests
                    for int_model in range(model_test_count):
                        if int_model != model_test_count-1:
                            xt, yt = x_test[samples*int_model:samples*(int_model+1)].compute(), y_test[samples*int_model:samples*(int_model+1)].compute()
                        else:
                            xt, yt = x_test[samples*int_model:].compute(), y_test[samples*int_model:].compute()
                    # Iterating through the different models
                        for int_cls, (cls_name, cls) in enumerate(partial_fit_classifiers.items()):
                        # Using the client to parallelize the fits
                            with joblib.parallel_backend('dask'):
                            # update estimator with data
                                acc[int_cls][int_model] = cls.score(xt, yt)
                                if cls_name == 'SGD' or cls_name == 'Perceptron' or cls_name == 'Passive-Aggressive':
                                    pred[cls_name].append(cls.decision_function(xt))
                                else:
                                    pred[cls_name].append(cls.predict_proba(xt))
                # Creating labels for AUROC
                    for int_cls, (cls_name, cls) in enumerate(partial_fit_classifiers.items()):
                    # model fitting stats
                        results.loc[cv,'%s %s %s Individual Train Time' % (error, reads, cls_name)] = np.mean(fit_time[int_cls])
                        results.loc[cv,'%s %s %s Individual Train Sample Number' % (error, reads, cls_name)] = len(y_train)
                    # accumulate test accuracy stats
                        results.loc[cv,'%s %s %s Individual Train Accuracy' % (error, reads, cls_name)] = np.mean(acc[int_cls])
                    # Calculating Confusion Matrix
                        conf_mat = confusion_matrix(y_test.compute(), np.concatenate(pred[cls_name]).argmax(axis=1))
                        np.save(os.path.join(pred_arrays, '%s %s %s Plasmid Prediction Array Train %s.npy' % (int(error*100), reads, cls_name, cv)), np.concatenate(pred[cls_name]))
                        np.save(os.path.join(pred_arrays, '%s %s %s Plasmid Confusion Matrix Train %s.npy' % (int(error*100), reads, cls_name, cv)), conf_mat)
                        pred[cls_name] = []
                # Printing the time the model testing has been running
                    print('Train set tested. Total time model testing has been running %s minutes' % ((time.perf_counter()-tick)/60))
                # Creating the test variables
                    acc_t = [np.zeros(model_unseen_count) for ii in partial_fit_classifiers]
                    test_time = [np.zeros(model_unseen_count) for ii in partial_fit_classifiers]
                # Iterating through the # of tests
                    for int_model in range(model_unseen_count):
                        if int_model != model_unseen_count-1:
                            xu, yu = unseen_data[samples*int_model:samples*(int_model+1)].compute(), y_unseen_data[samples*int_model:samples*(int_model+1)].compute()
                        else:
                            xu, yu = unseen_data[samples*int_model:].compute(), y_unseen_data[samples*int_model:].compute()
                    # Iterating through the different models
                        for int_cls, (cls_name, cls) in enumerate(partial_fit_classifiers.items()):
                        # timing each classifier
                            tick1 = time.perf_counter()
                        # Using the client to parallelize the fits
                            with joblib.parallel_backend('dask'):
                            # update estimator with data
                                acc_t[int_cls][int_model] = cls.score(xu, yu)
                                if cls_name == 'SGD' or cls_name == 'Perceptron' or cls_name == 'Passive-Aggressive':
                                    pred[cls_name].append(cls.decision_function(xu))
                                else:
                                    pred[cls_name].append(cls.predict_proba(xu))
                            test_time[int_cls][int_model] = time.perf_counter() - tick1
                # Creating labels for AUROC
                    for int_cls, (cls_name, cls) in enumerate(partial_fit_classifiers.items()):
                    # model fitting stats
                        results.loc[cv,'%s %s %s Individual Test Time' % (error, reads, cls_name)] = np.mean(test_time[int_cls])
                        results.loc[cv,'%s %s %s Individual Test Number' % (error, reads, cls_name)] = len(y_unseen_data)
                    # accumulate test accuracy stats
                        results.loc[cv,'%s %s %s Individual Test Accuracy' % (error, reads, cls_name)] = np.mean(acc_t[int_cls])
                    # Calculating Confusion Matrix
                        conf_mat = confusion_matrix(y_unseen_data.compute(), np.concatenate(pred[cls_name]).argmax(axis=1))
                        np.save(os.path.join(pred_arrays, '%s %s %s Plasmid Prediction Array Test %s.npy' % (int(error*100), reads, cls_name, cv)), np.concatenate(pred[cls_name]))
                        np.save(os.path.join(pred_arrays, '%s %s %s Plasmid Confusion Matrix Test %s.npy' % (int(error*100), reads, cls_name, cv)), conf_mat)
                        pred[cls_name] = []
                    print('Took %s minutes to fit, validate and test all models for 1 cross-fold validation' % ((time.perf_counter()-tick)/60))
                    print('%s fold Cross Validation finished' % (cv))

            # Timing how long it takes to run through all discriminant analysis models
                tick0 = time.perf_counter()
            ## Creating component analysis data on the first 500 SERS sequences out of the 1000
                x_train, x_test, y_train, y_test = train_test_split(train_data, y_train_data, random_state=123)
                xt, yt = x_test.compute(), y_test.compute()
                x, y = x_train[:int(samples-len(xt))].compute(), y_train[:int(samples-len(yt))].compute()
                xu, yu = unseen_data[:samples].compute(), y_unseen_data[:samples].compute()
                classifier = {
                        'PCA_svd': PCA(random_state=123),
                        'LDA': LinearDiscriminantAnalysis(solver='svd'),
                        'QDA': QuadraticDiscriminantAnalysis()
                }
                for cls_name, cls in classifier.items():
                    if cls_name == 'PCA_svd':
                        pi_pca_train = np.array(cls.fit_transform(np.concatenate((x,xt),axis=0)))
                        pi_pca_unseen = np.array(cls.transform(xu))
                        pi_pca = np.concatenate((pi_pca_train,pi_pca_unseen),axis=0)
                        pi_name = np.concatenate((y,yt,yu)).astype('int8')
                        pi_var = np.array([.32,.32,.36])
                        np.save(os.path.join(pred_arrays, '%s %s Plasmid PCA.npy' % (int(100*error), reads)), pi_pca)
                        pca_plot(pi_pca[:,:2],pi_name,'PI',int(100*error),reads)
                    else:
                    # fit model
                        cls.fit(x,y)
                    # CV Testing
                        results.loc[0,'%s %s %s Individual Train Accuracy' % (error, reads, cls_name)] = np.mean(cls.score(xt,yt))
                        pred[cls_name].append(cls.predict_proba(xt))
                    # Calculating Confusion Matrix
                        conf_mat = confusion_matrix(yt, np.concatenate(pred[cls_name]).argmax(axis=1))
                        np.save(os.path.join(pred_arrays, '%s %s %s Plasmid Prediction Array Train %s.npy' % (int(error*100), reads, cls_name, cv)), np.concatenate(pred[cls_name]))
                        np.save(os.path.join(pred_arrays, '%s %s %s Plasmid Confusion Matrix Train %s.npy' % (int(error*100), reads, cls_name, cv)), conf_mat)
                        pred[cls_name] = []
                    # Unknown Testing
                        results.loc[0,'%s %s %s Individual Test Accuracy' % (error, reads, cls_name)] = np.mean(cls.score(xu,yu))
                        pred[cls_name].append(cls.predict_proba(xu))
                    # Calculating Confusion Matrix
                        conf_mat = confusion_matrix(yu, np.concatenate(pred[cls_name]).argmax(axis=1))
                        np.save(os.path.join(pred_arrays, '%s %s %s Plasmid Prediction Array Test %s.npy' % (int(error*100), reads, cls_name, cv)), np.concatenate(pred[cls_name]))
                        np.save(os.path.join(pred_arrays, '%s %s %s Plasmid Confusion Matrix Test %s.npy' % (int(error*100), reads, cls_name, cv)), conf_mat)
                        pred[cls_name] = []
                print('Took %s minutes to fit, validate and test all discriminant analysis models' % ((time.perf_counter()-tick0)/60))
                print('Took %s hours to run all models' % ((time.perf_counter()-tick)/3600))

            #Saving the results
                results.to_csv('%s_ML_Results_%s_%s_%s.csv' % (file_type,reads,int(100*error),int_data))
                print('Results saved')
                print(datetime.datetime.now().isoformat())
                
            # Getting the # of times the group model needs to be iterated on
                if len(unseen_group_data) % samples != 0:
                    model_unseen_count = int(len(unseen_group_data)/samples) + 1
                else:
                    model_unseen_count = int(len(unseen_group_data)/samples)
                
            # Creating the split for the 90% train, 10% validate split for the training set
                for cv in range(cfv):
                # Here are the classifiers that support the `partial_fit` method
                    partial_fit_classifiers = {
                        'SGD': SGDClassifier(random_state=123),
                        'Perceptron': Perceptron(tol=1e-3,random_state=123),
                        'Passive-Aggressive': PassiveAggressiveClassifier(tol=1e-3,random_state=123),
                        'Neural Network': MLPClassifier(random_state=123),
                        'GNB': GaussianNB(),
                        'BNB': BernoulliNB(),
                        'RF': RandomForestClassifier(n_estimators=10,max_features=None,bootstrap=False,random_state=123,warm_start=True),
                        'ET': ExtraTreesClassifier(n_estimators=10,max_features=None,bootstrap=False,random_state=123,warm_start=True),
                        'GB': GradientBoostingClassifier(n_estimators=10,max_features=None,random_state=123,warm_start=True)
                    }
                # Timing how long it takes to run through all models
                    tick = time.perf_counter()
                # Getting the split for this cv
                    x_train, x_test, y_train, y_test = train_test_split(train_group_data, y_train_group, random_state=123+cv)
                # Getting the # of times the model needs to be iterated on
                    if len(x_train) % samples != 0:
                        model_fit_count = int(len(x_train)/samples) + 1
                    else:
                        model_fit_count = int(len(x_train)/samples)
                    if len(x_test) % samples != 0:
                        model_test_count = int(len(x_test)/samples) + 1
                    else:
                        model_test_count = int(len(x_test)/samples)
                # Iterating through the # of fits
                    for int_model in range(model_fit_count): 
                    # Getting the values for the model for testing
                        if int_model != model_fit_count-1:
                            x, y = x_train[samples*int_model:samples*(int_model+1)].compute(), y_train[samples*int_model:samples*(int_model+1)].compute()
                        else:
                            x, y = x_train[samples*int_model:].compute(), y_train[samples*int_model:].compute()
                    # Creating the timing variable
                        fit_time = [np.zeros(model_fit_count) for ii in partial_fit_classifiers]
                    # Iterating through the different models
                        for int_cls, (cls_name, cls) in enumerate(partial_fit_classifiers.items()):
                        # timing each classifier
                            tick1 = time.perf_counter()
                        # Using the client to parallelize the fits
                            with joblib.parallel_backend('dask'):
                            # update estimator with data
                                if cls_name == 'RF' or cls_name == 'ET' or cls_name == 'GB':
                                    cls.fit(x, y)
                                    if int_model != model_fit_count-1:
                                        cls.n_estimators += 1
                                else:
                                    cls.partial_fit(x, y, classes=[0,1])
                            fit_time[int_cls][int_model] = time.perf_counter() - tick1
                # Printing the time it took to fit all of the models
                    print('Took %s minutes to fit all models' % ((time.perf_counter()-tick)/60))
                # Creating the test variables
                    acc = [np.zeros(model_test_count) for ii in range(len(partial_fit_classifiers))]
                # Iterating through the # of tests
                    for int_model in range(model_test_count):
                        if int_model != model_test_count-1:
                            xt, yt = x_test[samples*int_model:samples*(int_model+1)].compute(), y_test[samples*int_model:samples*(int_model+1)].compute()
                        else:
                            xt, yt = x_test[samples*int_model:].compute(), y_test[samples*int_model:].compute()
                    # Iterating through the different models
                        for int_cls, (cls_name, cls) in enumerate(partial_fit_classifiers.items()):
                        # Using the client to parallelize the fits
                            with joblib.parallel_backend('dask'):
                            # update estimator with data
                                acc[int_cls][int_model] = cls.score(xt, yt)
                                if cls_name == 'SGD' or cls_name == 'Perceptron' or cls_name == 'Passive-Aggressive':
                                    pred[cls_name].append(cls.decision_function(xt))
                                else:
                                    pred[cls_name].append(cls.predict_proba(xt))
                # Creating labels for AUROC
                    for int_cls, (cls_name, cls) in enumerate(partial_fit_classifiers.items()):
                    # model fitting stats
                        results_group.loc[cv,'%s %s %s Group Train Time' % (error, reads, cls_name)] = np.mean(fit_time[int_cls])
                        results_group.loc[cv,'%s %s %s Group Train Sample Number' % (error, reads, cls_name)] = len(y_train)
                    # accumulate test accuracy stats
                        results_group.loc[cv,'%s %s %s Group Train Accuracy' % (error, reads, cls_name)] = np.mean(acc[int_cls])
                    # Calculating Confusion Matrix
                        if cls_name == 'SGD' or cls_name == 'Perceptron' or cls_name == 'Passive-Aggressive':
                            conf_mat = confusion_matrix(y_test.compute(), np.where(np.concatenate(pred[cls_name])>0,1,0))
                        else:
                            conf_mat = confusion_matrix(y_test.compute(), np.concatenate(pred[cls_name]).argmax(axis=1))
                        np.save(os.path.join(pred_arrays, '%s %s %s Plasmid Group Prediction Array Train %s.npy' % (int(error*100), reads, cls_name, cv)), np.concatenate(pred[cls_name]))
                        np.save(os.path.join(pred_arrays, '%s %s %s Plasmid Group Confusion Matrix Train %s.npy' % (int(error*100), reads, cls_name, cv)), conf_mat)
                        pred[cls_name] = []
                # Printing the time the model testing has been running
                    print('Train set tested. Total time model testing has been running %s minutes' % ((time.perf_counter()-tick)/60))
                # Creating the test variables
                    acc_t = [np.zeros(model_unseen_count) for ii in partial_fit_classifiers]
                    test_time = [np.zeros(model_unseen_count) for ii in partial_fit_classifiers]
                # Iterating through the # of tests
                    for int_model in range(model_unseen_count):
                        if int_model != model_unseen_count-1:
                            xu, yu = unseen_group_data[samples*int_model:samples*(int_model+1)].compute(), y_unseen_group[samples*int_model:samples*(int_model+1)].compute()
                        else:
                            xu, yu = unseen_group_data[samples*int_model:].compute(), y_unseen_group[samples*int_model:].compute()
                    # Iterating through the different models
                        for int_cls, (cls_name, cls) in enumerate(partial_fit_classifiers.items()):
                        # timing each classifier
                            tick1 = time.perf_counter()
                        # Using the client to parallelize the fits
                            with joblib.parallel_backend('dask'):
                            # update estimator with data
                                acc_t[int_cls][int_model] = cls.score(xu, yu)
                                if cls_name == 'SGD' or cls_name == 'Perceptron' or cls_name == 'Passive-Aggressive':
                                    pred[cls_name].append(cls.decision_function(xu))
                                else:
                                    pred[cls_name].append(cls.predict_proba(xu))
                            test_time[int_cls][int_model] = time.perf_counter() - tick1
                # Creating labels for AUROC
                    for int_cls, (cls_name, cls) in enumerate(partial_fit_classifiers.items()):
                    # model fitting stats
                        results_group.loc[cv,'%s %s %s Group Test Time' % (error, reads, cls_name)] = np.mean(test_time[int_cls])
                        results_group.loc[cv,'%s %s %s Group Test Number' % (error, reads, cls_name)] = len(y_unseen_group)
                    # accumulate test accuracy stats
                        results_group.loc[cv,'%s %s %s Group Test Accuracy' % (error, reads, cls_name)] = np.mean(acc_t[int_cls])
                    # Calculating Confusion Matrix
                        if cls_name == 'SGD' or cls_name == 'Perceptron' or cls_name == 'Passive-Aggressive':
                            conf_mat = confusion_matrix(y_unseen_group.compute(), np.where(np.concatenate(pred[cls_name])>0,1,0))
                        else:
                            conf_mat = confusion_matrix(y_unseen_group.compute(), np.concatenate(pred[cls_name]).argmax(axis=1))
                        np.save(os.path.join(pred_arrays, '%s %s %s Plasmid Group Prediction Array Test %s.npy' % (int(error*100), reads, cls_name, cv)), np.concatenate(pred[cls_name]))
                        np.save(os.path.join(pred_arrays, '%s %s %s Plasmid Group Confusion Matrix Test %s.npy' % (int(error*100), reads, cls_name, cv)), conf_mat)
                        pred[cls_name] = []
                    print('Took %s minutes to fit, validate and test all models for 1 cross-fold validation' % ((time.perf_counter()-tick)/60))
                    print('%s fold Cross Validation finished' % (cv))
					
            # Timing how long it takes to run through all discriminant analysis models
                tick0 = time.perf_counter()
            ## Creating component analysis data on the first 500 SERS sequences out of the 1000
                x_train, x_test, y_train, y_test = train_test_split(train_group_data, y_train_group, random_state=123)
                x, y = x_train[:samples].compute(), y_train[:samples].compute()
                xt, yt = x_test.compute(), y_test.compute()
                xu, yu = unseen_group_data[:samples].compute(), y_unseen_group[:samples].compute()
                classifier = {
                        'PCA_svd': PCA(random_state=123),
                        'LDA': LinearDiscriminantAnalysis(solver='svd'),
                        'QDA': QuadraticDiscriminantAnalysis()
                }
                for cls_name, cls in classifier.items():
                    if cls_name == 'PCA_svd':
                        pg_pca_train = np.array(cls.fit_transform(np.concatenate((x,xt),axis=0)))
                        pg_pca_unseen = np.array(cls.transform(xu))
                        pg_pca = np.concatenate((pg_pca_train,pg_pca_unseen),axis=0)
                        pg_name = np.concatenate((y,yt,yu)).astype('int8')
                        pg_var = np.array([.32,.32,.36])
                        np.save(os.path.join(pred_arrays, '%s %s Plasmid Group PCA.npy' % (int(100*error), reads)), pg_pca)
                        pca_plot(pg_pca[:,:2],pg_name,'PG',int(100*error),reads)
                    else:
                    # fit model
                        cls.fit(x,y)
                    # CV Testing
                        results_group.loc[0,'%s %s %s Group Train Accuracy' % (error, reads, cls_name)] = np.mean(cls.score(xt,yt))
                        pred[cls_name].append(cls.predict_proba(xt))
                    # Calculating Confusion Matrix
                        conf_mat = confusion_matrix(yt, np.concatenate(pred[cls_name]).argmax(axis=1))
                        np.save(os.path.join(pred_arrays, '%s %s %s Plasmid Group Prediction Array Test %s.npy' % (int(error*100), reads, cls_name, cv)), np.concatenate(pred[cls_name]))
                        np.save(os.path.join(pred_arrays, '%s %s %s Plasmid Group Confusion Matrix Test %s.npy' % (int(error*100), reads, cls_name, cv)), conf_mat)
                        pred[cls_name] = []
                    # Unknown Testing
                        results_group.loc[0,'%s %s %s Group Test Accuracy' % (error, reads, cls_name)] = np.mean(cls.score(xu,yu))
                        pred[cls_name].append(cls.predict_proba(xu))
                    # Calculating Confusion Matrix
                        conf_mat = confusion_matrix(yu, np.concatenate(pred[cls_name]).argmax(axis=1))
                        np.save(os.path.join(pred_arrays, '%s %s %s Plasmid Group Prediction Array Test %s.npy' % (int(error*100), reads, cls_name, cv)), np.concatenate(pred[cls_name]))
                        np.save(os.path.join(pred_arrays, '%s %s %s Plasmid Group Confusion Matrix Test %s.npy' % (int(error*100), reads, cls_name, cv)), conf_mat)
                        pred[cls_name] = []
                print('Took %s minutes to fit, validate and test all discriminant analysis models' % ((time.perf_counter()-tick0)/60))
                print('Took %s hours to run Individual and Group models' % ((time.perf_counter()-tick)/3600))

            #Saving the results
                results_group.to_csv('Plasmid_ML_Group_Results_%s_%s_%s.csv' % (reads,int(100*error),int_data))
                print('Results saved')
                print(datetime.datetime.now().isoformat())

                print('%s error rate and %s number of reads tests finished' % (error, reads))
            print('Took %s hours to run all number of reads tests' % ((time.perf_counter()-start1)/3600))
        print('Took %s days to run all error rate tests' % ((time.perf_counter()-start)/(3600*24)))           
         
client.close()
cluster.close()

print(begin)
print(datetime.datetime.now().isoformat())
