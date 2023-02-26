import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

catalogdir = '/Users/toneill/Box/MC/HST/'
photdir = '/Users/toneill/N159/photometry/'

#full = pd.read_csv(photdir + 'dered_hstphot_all.csv')
train_full = pd.read_csv(catalogdir + 'Ksoll2018_training_set.csv')



obs0 = train_full[['m_f336w','m_f555w', 'm_f658n','m_f775w',
       'm_f110w','m_f160w']].iloc[4]

bands = [3360,5550,6580,7750,11000,16000]

plt.figure()
plt.plot(bands,obs0.values,marker='o',c='grey',ls='--', drawstyle='steps-post')
#plt.gca().set_yscale('log')
plt.gca().invert_yaxis()
plt.ylabel('mag')
plt.xlabel('angstrom')

# m = -2.5 log10 (F_obj / F_vega)
# (-1/2.5) m = log10()
# 10^((-1/2.5) m) = F_obj/F_vega

fluxes = 10**((-1/2.5)*obs0)

plt.plot(bands,fluxes,marker='o',c='grey',ls='--', drawstyle='steps-post')
plt.gca().set_yscale('log')








#https://pysynphot.readthedocs.io/en/latest/index.html#installation-and-setup
#https://pysynphot.readthedocs.io/en/latest/bandpass.html

import pysynphot as S

bp_555 = S.ObsBandpass('acs,wfc1,f555w')
bp_775 = S.ObsBandpass('acs,wfc1,f775w')
bp_814 = S.ObsBandpass('acs,wfc1,f814w')
bp_110 = S.ObsBandpass('wfc3,ir,f110w')


flat_555 = S.FlatSpectrum(obs0['m_f555w'],fluxunits='vegamag')
flat_775 = S.FlatSpectrum(obs0['m_f775w'],fluxunits='vegamag')
flat_110 = S.FlatSpectrum(obs0['m_f110w'],fluxunits='vegamag')

mult = flat_555*bp_555 + flat_775*bp_775 + flat_110*bp_110

plt.plot(mult.wave, mult.flux, drawstyle='steps-mid',ls='--')
plt.gca().invert_yaxis()
plt.gca().set_xscale('log')
plt.xlabel(mult.waveunits)
plt.ylabel(mult.fluxunits)

obs_mult = S.Observation(mult,bp_814)
plt.plot(obs_mult.binwave, obs_mult.binflux, drawstyle='steps-mid')
plt.plot(bp_814.binset, bp_814(bp_814.binset)*200, drawstyle='steps-mid')


obs_555 = S.Observation(flat_555,bp_555)
obs_775 = S.Observation(flat_775,bp_775)
obs_110 = S.Observation(flat_110,bp_110)

plt.figure()
plt.plot(obs_555.binwave, obs_555.binflux, drawstyle='steps-mid')
plt.plot(obs_775.binwave, obs_775.binflux, drawstyle='steps-mid')
plt.plot(obs_110.binwave, obs_110.binflux, drawstyle='steps-mid')
plt.gca().invert_yaxis()
plt.gca().set_xscale('log')




bp = S.ObsBandpass('acs,wfc1,f775w')
other_bp = S.ObsBandpass('acs,wfc1,f814w')
bp.check_overlap(other_bp)


obs = S.Observation(
       S.BlackBody(5000).renorm(18.6, 'vegamag', S.ObsBandpass('i')),
       S.ObsBandpass('band(wfpc,f555w)'))

mag = obs.effstim('vegamag')

plt.figure()
plt.plot(obs.binwave, obs.binflux, drawstyle='steps-mid')


bp_555.wave
#http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?id=HST/ACS_WFC.F555W&&mode=browse&gname=HST&gname2=ACS_WFC#filter
box_555 = S.Box(bp_555.avgwave(), bp_555.rectwidth())
       #.renorm(20,'vegamag',S.ObsBandpass('f555w'))

plt.plot(box_555.wave, box_555.throughput, 'b')#,

####################################################

sp_555 = S.ArraySpectrum(bp_555.wave,np.repeat(1,len(bp_555.wave)))
sp_555 = sp_555.renorm(obs0['m_f555w'],'vegamag',bp_555)
obs_555 = S.Observation(sp_555,bp_555)

plt.plot(obs_555.wave, obs_555.flux, drawstyle='steps-mid')
plt.ylabel(obs_555.fluxunits)
mag_555 = obs_555.effstim('vegamag')
mag_555

sp_775 = S.ArraySpectrum(bp_775.wave,np.repeat(1,len(bp_775.wave)))
sp_775 = sp_775.renorm(obs0['m_f775w'],'vegamag',bp_775)
obs_775 = S.Observation(sp_775,bp_775)
mag_775 = obs_775.effstim('vegamag')
mag_775

sp_110 = S.ArraySpectrum(bp_775.wave,np.repeat(1,len(bp_775.wave)))
sp_110 = sp_110.renorm(obs0['m_f110w'],'vegamag',bp_110)
obs_110 = S.Observation(sp_110,bp_110)
mag_110 = obs_110.effstim('vegamag')
mag_110
#obs0['m_f110w']


manflux_mult = obs_775.flux + obs_555.flux#sp_110.flux +
man_sp = S.ArraySpectrum(bp_775.wave,manflux_mult)
man_obs_555 = S.Observation(man_sp,bp_555)
man_mag_555 = man_obs_555.effstim('vegamag')
man_mag_555
obs0['m_f555w']

mult = sp_555*bp_555 + sp_775*bp_775 + sp_110*bp_110
mult2 = mult.renorm(obs0['m_f775w'],'vegamag',bp_775)

#mult = sp_555 + sp_775 + sp_110

mult_obs_775 = S.Observation(mult2,bp_775)
mult_mag_775 = mult_obs_775.effstim('vegamag')
print(mult_mag_775)
print(obs0['m_f775w'])

mult_obs_110 = S.Observation(mult2,bp_110)
mult_mag_110 = mult_obs_110.effstim('vegamag')
print(mult_mag_110)
print(obs0['m_f110w'])


plt.figure()
plt.plot(obs_555.binwave, obs_555.binflux, drawstyle='steps-mid')
plt.plot(obs_775.binwave, obs_775.binflux, drawstyle='steps-mid')
plt.plot(obs_110.binwave, obs_110.binflux, drawstyle='steps-mid')
plt.plot(mult.wave, mult.flux, drawstyle='steps-mid',ls=':',c='k')
#plt.plot(mult2.wave, mult2.flux, drawstyle='steps-mid',ls='--',c='r')
plt.plot(mult_obs_775.binwave, mult_obs_775.binflux, drawstyle='steps-mid')
plt.plot(mult_obs_110.binwave, mult_obs_110.binflux, drawstyle='steps-mid')
plt.ylabel(obs_775.fluxunits)


plt.figure()
plt.plot(sp_555.wave, sp_555.flux, drawstyle='steps-mid')
plt.plot(sp_775.wave, sp_775.wave, drawstyle='steps-mid')
plt.plot(sp_110.wave, sp_110.wave, drawstyle='steps-mid')
plt.gca().set_yscale('log')







#bp_v = S.ObsBandpass('johnson,v')
plt.plot(bp_555.binset, bp_555(bp_555.binset), 'b')#,
       bp_v.wave, bp_v.throughput, 'g--')
plt.xlim(4000, 7000)
plt.xlabel(bp_555.waveunits)
plt.ylabel('throughput')
plt.legend([bp_555.name, 'Johnson V'], loc='best')