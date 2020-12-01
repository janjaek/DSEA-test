import pandas as pd
import numpy as np
from scipy.stats import normaltest
from sklearn.preprocessing import power_transform
from mord import LogisticAT
from matplotlib import pyplot as plt
from sklearn.feature_selection import RFE
df=pd.read_csv('sample.csv',nrows=4000000)
Y = df['MCPrimary.energy']
binning_E = np.logspace(np.log10(Y.min()-10),
np.log10(Y.max()+100), 13)    # creating 12 bins for the energy 



#Binning overflow & underflow manually
overflow = 1e10
bins_over = np.append(binning_E,overflow)
binning_E = bins_over

Y = np.digitize(Y, binning_E)  

Y = Y - 1

n_test = int(len(df)/10)
Y_train = Y[n_test:]
Y_test = Y[:n_test]
X = df[['LineFitGeoSplit1Params.n_hits',
        'SplineMPEDirectHitsICB.n_early_strings',
        'SplineMPEDirectHitsICB.n_late_doms',
        'SPEFitSingleTimeSplit1.azimuth',
        'ProjectedQ.max_grad_radius_circ_F',
        'ProjectedQ.ratio',
        'BestTrackCramerRaoParams.cramer_rao_theta',
        'BestTrackCramerRaoParams.variance_theta',
        'BestTrackCramerRaoParams.variance_x',
        'BestTrackCramerRaoParams.variance_y',
        'BestTrackCramerRaoParams.covariance_theta_y',
        'SplineMPETruncatedEnergy_SPICEMie_DOMS_Muon.energy',
        'SplineMPETruncatedEnergy_SPICEMie_BINS_Muon.energy',
        'SPEFit2TimeSplit1BayesianFitParams.nmini',
        'LineFitTimeSplit2Params.n_hits',
        'BestTrackDirectHitsICB.n_dir_pulses',
        'HitStatisticsValues.min_pulse_time',
        'SplineMPEDirectHitsICE.n_dir_doms',
        'SplineMPEDirectHitsICE.n_late_strings',
        'MPEFit_HVFitParams.nmini']]
        #'SplineMPECharacteristicsIC.avg_dom_dist_q_tot_dom',
        #'MPEFitHighNoiseFitParams.nmini']]
X_box= power_transform(X, method='yeo-johnson')    
X_btrain = X_box[n_test:]#splitting the dataframe
X_btest = X_box[:n_test]
estimator= LogisticAT()
selector = RFE(estimator, n_features_to_select=5, step=1)
selector.fit(X_box,Y)
print(selector.ranking_)