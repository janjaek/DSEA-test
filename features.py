import pandas as pd
import numpy as np
from scipy.stats import normaltest

from matplotlib import pyplot as plt
df=pd.read_csv('sample.csv',nrows=1000000) #read the sample data

from sklearn.preprocessing import power_transform #import yeo-johnson transformation
df=df.drop(['MCPrimary.energy'],axis=1)# drop the target 
X_t=power_transform(df, method='yeo-johnson') # transform data to make it gaußian
stats, p = normaltest(X_t)# looking at skewness and kurtosis
print(p)