#!/usr/bin/env python

"""
Pre-processing code for the KDD-1999 dataset, using Pandas
"""

import pandas as pd
import csv

def preprocess(inFile,outFile):
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
    colsToTransform = [
                        namesVec[1],
                        namesVec[2],
                        namesVec[3],
                        namesVec[6],
                        namesVec[11],
                        namesVec[20],
                        namesVec[21],
                        namesVec[41]
                      ]
    print('Transforming Data')
    for col in colsToTransform:
        x[col], label_mapping[col] = pd.factorize(x[col].astype('category'))

    # write the data back out
    print('Writing Data')
    x.to_csv(outFile)

    f = open(outFile+'.mapping','wb')
    w = csv.DictWriter(f,label_mapping.keys())
    w.writerow(label_mapping)
    f.close()


if __name__=='__main__':
    inFile = '/Users/kiran/Documents/data/kdd1999/kddcup.data'
    outFile = '/Users/kiran/Documents/data/kdd1999/kddcup.process.data'
    preprocess(inFile,outFile)