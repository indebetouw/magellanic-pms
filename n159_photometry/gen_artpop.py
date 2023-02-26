
import artpop
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy import units as u

#artpop.get_filter_names('HST_WFC3')
filts_use = ['WFC3_UVIS_F555W','WFC3_UVIS_F775W',
             'WFC3_UVIS_F814W','WFC3_IR_F110W',
             'WFC3_IR_F125W','WFC3_IR_F160W']

popprops = {'feh':-0.30, 'phot_system':'HST_WFC3',
            'imf':'kroupa','distance':50*u.kpc,
            'num_stars':1e4,'mag_limit':27,'mag_limit_band':'WFC3_UVIS_F555W'}#,'a_lam':0.3}

age_range = np.logspace(np.log10(5e5),7,10)
#age_range = np.append(age_range,10**8)

ssp = artpop.MISTSSP( log_age = np.log10(age_range[0]),  **popprops)
phases = ssp.get_star_phases()
for age in age_range[1::]:
    ssp_a = artpop.MISTSSP(log_age=np.log10(age),**popprops)
    ssp = ssp + ssp_a
    phases = np.append(phases, ssp_a.get_star_phases())

#ssp.mag_table

def add_noise(mag,mu=1,sigma=1):
    #noise = np.random.normal(mu,sigma,len(mag))
    return mag + np.random.normal(mu,sigma,len(mag))


m555 = ssp.star_mags(filts_use[0])
m775 = ssp.star_mags(filts_use[1])
m814 = ssp.star_mags(filts_use[2])
m110 = ssp.star_mags(filts_use[3])
m125 = ssp.star_mags(filts_use[4])
m160 = ssp.star_mags(filts_use[5])
col_57 = m555 - m775
col_58 = m555 - m814

mu = 0
sigma = 0.15
mag5 = add_noise(m555,mu=mu,sigma=sigma)
mag7 = add_noise(m775,mu=mu,sigma=sigma)
mag8 = add_noise(m814,mu=mu,sigma=sigma)
mag1 = add_noise(m110,mu=mu,sigma=sigma)
mag2 = add_noise(m125,mu=mu,sigma=sigma)
mag6 = add_noise(m160,mu=mu,sigma=sigma)

c_57 = mag5 - mag7
c_58 = mag5 - mag8

#agegroup = ssp.ssp_labels
#age_range
#agedict = [{i+1:np.round(age_range[i])} for i in range(len(age_range))]
#agedict
#ages = agedict[agegroup]
#[agedict[agegroup[i]] for i in range(len(agegroup))]


popdf = pd.DataFrame({'phase':phases,'agegroup':ssp.ssp_labels,
                      'F555Wmag':m555,'F775Wmag':m775,'F814Wmag':m814,
                      'F110Wmag':m110,'F125Wmag':m125,'F160Wmag':m160})
popdf.to_csv('/Users/toneill/N159/isochrones/artpop_df.csv',index=False)#_noisy

plt.figure(figsize=(6,8))
#plt.scatter(c_57,mag5,c=ssp.ssp_labels,s=0.2,alpha=0.9)
#plt.scatter(col_57,m555,c='k',s=0.05,marker='x')
plt.scatter(col_57,m555,c=ssp.ssp_labels,s=0.05,marker='x')
plt.gca().invert_yaxis()
plt.xlabel('555 - 775')
plt.ylabel('555')
plt.tight_layout()




# generate boolean phase masks (ONLY WORKS IF SINGLE SSP)
#PMS = ssp.select_phase('PMS')
# or could single out which age came from via
# old = ssp.ssp_labels == 1
# etc
unique_phases = np.unique(phases)
plt.figure(figsize=(6,7))
for phase in unique_phases:
    pphase = phases == phase
    plt.scatter(col_57[pphase],m555[pphase], s=3, label=phase)
plt.legend()
plt.gca().invert_yaxis()
plt.tight_layout()
plt.xlabel('555 - 775')
plt.ylabel('555')


########################################################################################
## using results of r regression
########################################################################################

# F814 = c0 + c1*F775 + c2*F110
trans_814 = -0.032 + 0.917*m775 + 0.247*m110 - 0.106*m160 -0.056*m555
# F125 = c0 + c1*F160 + c2*F110
trans_125 = 0.033 -0.120*m775 + 0.884*m110 + 0.207*m160 + 0.027*m555

plt.figure()
plt.scatter(trans_814,m814-trans_814,label='F814W residuals',s=3)
plt.scatter(trans_125,m125-trans_125,label='F125W residuals',s=3)
plt.axhline(y=0,c='k',ls='--',lw=0.75)
plt.xlabel('fit value')
plt.ylabel('residual')
plt.legend()
plt.ylim(-0.1,0.1)


plt.figure()
plt.hist(trans_814/m775,label='trans 814',alpha=0.8,bins=30)
plt.hist(m814/m775,label='orig 814',alpha=0.8,bins=30)
plt.legend()
plt.axvline(x=1,c='k',ls='--',lw=0.75)
plt.xlabel('[ mag ] / m775')


############################################

plt.figure(figsize=(6,6))
plt.scatter(m775,m814,c='k',s=2)
#plt.gca().invert_yaxis()
plt.xlabel('775')
plt.ylabel('814')


import statsmodels.api as sm

x = m775
X = sm.add_constant(x)
y = m814
model = sm.OLS(y,X)
results = model.fit()
results.summary()

Xnew = np.linspace(np.min(x),np.max(x),1000)
ynewpred = results.params[1]*Xnew + results.params[0]

plt.figure()
plt.scatter(x,y,c='cornflowerblue',s=2)
plt.plot(Xnew,ynewpred,c='r',lw=3,ls='--')
plt.gca().invert_yaxis()

