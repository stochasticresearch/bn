#!/usr/bin/env python

import pandas as pd
import numpy as np
import scipy.io as sio

if __name__=='__main__':

    if platform == "linux" or platform == "linux2":
        folder = '/home/kiran/ownCloud/PhD/sim_results/crime/'
    elif platform == "darwin":
        folder = '/Users/Kiran/ownCloud/PhD/sim_results/crime/'
    elif platform == "win32":
        folder = 'C:\\Users\\kiran\\ownCloud\\PhD\\sim_results\\crime'
    
    f = os.path.join(folder,'communities.data')
    df = pd.read_csv(f,header=None)
    df = df.replace('?', np.NaN)

    # extract columns w/out missing data
    colsToExtract = np.setdiff1d(df.columns,df.columns[df.isnull().any()].tolist())

    df = df[colsToExtract]

    # extract only numeric data
    df = df.loc[:, df.dtypes != object]
    sio.savemat(os.path.join(folder,'communities.mat'), mdict={'X': df.values})