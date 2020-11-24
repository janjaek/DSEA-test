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
X = df[['SPEFit2GeoSplit2BayesianFitParams.logl',
        'SplineMPEDirectHitsICC.dir_track_hit_distribution_smoothness',
        'HitStatisticsValuesIC.q_max_doms',
        'HitStatisticsValuesIC.q_tot_pulses',
        'SPEFit2GeoSplit2FitParams.nmini',
        'HitMultiplicityValuesIC.n_hit_doms_one_pulse',
        'BestTrackDirectHitsICA.dir_track_hit_distribution_smoothness',
        'SplineMPETruncatedEnergy_SPICEMie_BINS_MuEres.value',
        'MPEFitParaboloidFitParams.err1',
        'MPEFitParaboloidFitParams.err2',
        'MPEFitParaboloidFitParams.rotang',
        'SplineMPEDirectHitsICA.dir_track_hit_distribution_smoothness',
        'SplineMPEDirectHitsICA.n_dir_pulses',
        'SPEFit2TimeSplit2BayesianFitParams.nmini',
        'SPEFit2GeoSplit1FitParams.nmini',
        'SplineMPEDirectHitsICD.n_dir_strings',
        'SplineMPE_SegementFitParams.rlogl',
        'SplineMPETruncatedEnergy_SPICEMie_BINS_Neutrino.energy',
        'SplineMPECharacteristicsIC.avg_dom_dist_q_tot_dom',
        'MPEFitHighNoiseFitParams.nmini']]
X_box= power_transform(X, method='yeo-johnson')    
X_btrain = X_box[n_test:]#splitting the dataframe
X_btest = X_box[:n_test]
estimator= LogisticAT()
selector = RFE(estimator, n_features_to_select=5, step=1)
selector.fit(X_box,Y)
print(selector.ranking_)