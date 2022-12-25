import pandas as pd

import scipy

from scipy import io

features_struct = scipy.io.loadmat('/Users/macbook/Desktop/骨质疏松数据/fusion_testflag_smote.mat')

features = features_struct['test_flag']

dfdata = pd.DataFrame(features)

datapath1 = './GZSS_label.csv'

dfdata.to_csv(datapath1, index=False)