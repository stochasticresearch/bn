#!/usr/bin/env python

"""
Pre-processing code for the KDD-1999 dataset, using Pandas
"""

import pandas as pd
import csv

def preprocess(inFile,outFile):
    numDiscreteCutOff = 3
    namesVec = [
                'duration', 
                'protocol_type',
                'service',
                'flag',
                'src_bytes',
                'dst_bytes',
                'land',
                'wrong_fragment',
                'urgent',
                'hot',
                'num_failed_logins',
                'logged_in',
                'num_compromised',
                'root_shell',
                'su_attempted',
                'num_root',
                'num_file_creations',
                'num_shells',
                'num_access_files',
                'num_outbound_cmds',
                'is_host_login',
                'is_guest_login',
                'count',
                'srv_count',
                'serror_rate',
                'srv_serror_rate',
                'rerror_rate',
                'srv_rerror_rate',
                'same_srv_rate',
                'diff_srv_rate',
                'srv_diff_host_rate',
                'dst_host_count',
                'dst_host_srv_count',
                'dst_host_same_srv_rate',
                'dst_host_diff_srv_rate',
                'dst_host_same_src_port_rate',
                'dst_host_srv_diff_host_rate',
                'dst_host_serror_rate',
                'dst_host_srv_serror_rate',
                'dst_host_rerror_rate',
                'dst_host_srv_rerror_rate',
                'class_label'
                ]
    print('Reading Data')
    x = pd.read_csv(inFile, names=namesVec)
    label_mapping = {}
    
    # TODO: get this from the .names file
    discreteIdxs = [1,2,3,6,11,20,21,41]
    colsToTransform = []
    for ii in range(len(discreteIdxs)):
        colsToTransform.append(namesVec[discreteIdxs[ii]])

    print('Transforming Data')
    for col in colsToTransform:
        x[col], label_mapping[col] = pd.factorize(x[col].astype('category'))

    # write the data back out after randomely sampling, b/c we have a
    # TON of data, and don't want to spend forever processing this data
    y_train = x.sample(n=10000, replace=False)
    y_test =  x.sample(n=10000, replace=False)

    # figure out which columns should be dropped because they only have 1 realization 
    # of a feature
    colsToDrop = []
    for col in range(len(x.columns)):
        if(len(y_train.ix[:,col].unique())<=numDiscreteCutOff or len(y_test.ix[:,col].unique())<=numDiscreteCutOff):
            colsToDrop.append(col)
    y_train.drop(y_train.columns[colsToDrop], axis=1, inplace=True)
    y_test.drop(y_test.columns[colsToDrop], axis=1, inplace=True)
    discreteIdxsDropped = list(discreteIdxs)
    for col in colsToDrop:
        if col in discreteIdxsDropped: discreteIdxsDropped.remove(col)
    colsToTransform = []
    for ii in range(len(discreteIdxsDropped)):
        colsToTransform.append(namesVec[discreteIdxsDropped[ii]])
    newDiscreteIdxsRemapped = []
    for ii in range(len(discreteIdxsDropped)):
        newDiscreteIdxsRemapped.append(y_train.columns.get_loc(namesVec[discreteIdxsDropped[ii]]))

    print('Original Discrete Idxs=' + str(discreteIdxs))
    print('Dropped=' + str(colsToDrop))
    print('Discrete Idxs after dropped=' + str(discreteIdxsDropped))
    print('New Discrete Idxs=' + str(newDiscreteIdxsRemapped))

    print('Writing Data')
    y_train.to_csv(outFile + '.train')
    y_test.to_csv(outFile + '.test')

    f = open(outFile+'.mapping','wb')
    w = csv.DictWriter(f,label_mapping.keys())
    w.writerow(label_mapping)
    f.close()

    f = open(outFile+'.discrete','w')
    for item in newDiscreteIdxsRemapped:
        f.write('%s\n' % str(item))
    f.close()

    f = open(outFile+'.colnames','w')
    colnames = y_train.columns.tolist()
    for item in colnames:
        f.write('%s\n' % str(item))
    f.close()

if __name__=='__main__':
    inFile = '/Users/kiran/Documents/data/kdd1999/kddcup.data'
    outFile = '/Users/kiran/Documents/data/kdd1999/kddcup.preprocess.data'
    preprocess(inFile,outFile)