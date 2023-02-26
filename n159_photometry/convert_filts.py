
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.table import Table

wdir = '/Users/toneill/N159/isochrones/'

i1 = 'hstacs.dat'
i2 = 'hstwfcwide.dat'


t1 = pd.read_csv(wdir+i1,skiprows=14,delimiter='\s+',skipfooter=1)#,dtype='float64')
t1cols = t1.columns
t1 = t1[t1cols[0:-1]]
t1.columns = t1cols[1::]
t1 = t1.iloc[0:-1]
t1cols = t1.columns
t1 = t1.astype('float')

t2 = pd.read_csv(wdir+i2,skiprows=14,delimiter='\s+',skipfooter=1)#,dtype='float64')
t2cols = t2.columns
t2 = t2[t2cols[0:-1]]
t2.columns = t2cols[1::]
t2 = t2.iloc[0:-1]
t2cols = t2.columns
t2 = t2.astype('float')


dmod = 18.48

use = (t2['logAge'] <= 7) & (t2['F555Wmag'] + dmod >= 15)

plt.figure(figsize=(6,8))
plt.scatter(t2['F555Wmag'][use]-t2['F814Wmag'][use],t2['F555Wmag'][use]+dmod,
            c=t2['Mass'][use])
plt.gca().invert_yaxis()


m_5 = t2['F555Wmag'][use]
m_7 = t2['F775Wmag'][use]
m_8 = t2['F814Wmag'][use]
m_1 = t2['F110Wmag'][use]
m_2 = t2['F125Wmag'][use]
m_6 = t2['F160Wmag'][use]

m_2 = t2['F125Wmag'][use]
col_58 = t2['F555Wmag'][use]-t2['F814Wmag'][use]
col_57 = t2['F555Wmag'][use]-t2['F775Wmag'][use]
col_16 = t2['F110Wmag'][use]-t2['F160Wmag'][use]
col_26 = t2['F125Wmag'][use]-t2['F160Wmag'][use]

plt.figure()
plt.scatter(col_57,m_8)

t2df = t2[use]

t2df['col58'] = col_58
t2df['col57'] = col_57
t2df['col16'] = col_16
t2df['col26'] = col_26

t2df.to_csv(wdir+f'{i2[0:-4]}.csv')

import statsmodels.api as sm

X = m_1
y = m_2
#X = sm.add_constant(X)
model = sm.OLS(y,X)
results = model.fit()
results.summary()

Xnew = np.linspace(np.min(X),np.max(X),1000)
#Xnew = sm.add_constant(Xnew)
ynewpred = results.params[0]*Xnew#results.params['const'] +
#results.predict(Xnew)  # predict out of sample

plt.figure()
plt.scatter(X,y,c='cornflowerblue',s=3)
plt.plot(Xnew,ynewpred,c='r',lw=3,ls='--')
plt.xlabel('110')
plt.ylabel('125')
plt.gca().invert_yaxis()
plt.title(f'R$^2$={results.rsquared:.2f}')

