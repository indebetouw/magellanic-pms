import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

catalogdir = '/Users/toneill/Box/MC/HST/'
photdir = '/Users/toneill/N159/photometry/'

full = pd.read_csv(photdir + 'dered_hstphot_all.csv')
train_full = pd.read_csv(catalogdir + 'Ksoll2018_training_set.csv')


#https://pysynphot.readthedocs.io/en/latest/index.html#installation-and-setup
#https://pysynphot.readthedocs.io/en/latest/bandpass.html

import pysynphot as S

import os
os.environ['PYSYN_CDBS']

bp_acs = S.ObsBandpass('acs,wfc1,f555w')
bp_v = S.ObsBandpass('johnson,v')
plt.plot(bp_acs.binset, bp_acs(bp_acs.binset), 'b',
       bp_v.wave, bp_v.throughput, 'g--')
plt.xlim(4000, 7000)
plt.xlabel(bp_acs.waveunits)
plt.ylabel('throughput')
plt.legend([bp_acs.name, 'Johnson V'], loc='best')