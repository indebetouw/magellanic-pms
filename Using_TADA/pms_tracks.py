#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 14:42:03 2021

@author: toneill
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

##############################
# Load data
##############################

# Load complete HST photometry
hst = pd.read_csv('schmalz2008_tab2_hst.txt',
                 skiprows=23,delim_whitespace=True,
                 names=['num','RAh','RAm','RAs',
                        'DEd','DEm','DEs',
                        'Vmag','e_Vmag',
                        'Imag','e_Imag'])
# Load HST data limited to PMS stars within contours drawn by RI
# on Goulermis+ 2012 CMD
PMS_df = pickle.load(open('pms_df.p','rb'))

# Load isochrones from TA-DA
pisa_phot = pd.read_csv('tada_pisa_syn_phot_expYrs.txt',
                        skiprows=18,delim_whitespace=True)
pisa_phot['555min814'] = pisa_phot['acswfc_f555w']-pisa_phot['acswfc_f814w']
pisa_phot = pisa_phot.dropna().reset_index(drop=True)

# divide by age of isochrone track and set up colors of tracks for plotting
unique_ages = np.unique(pisa_phot['log_age_yr'])
fin_colors = {unique_ages[0]:'hotpink', # 0.5 Myr
              unique_ages[1]:'r', # 1 Myr
              unique_ages[2]:'orange', # 1.5 Myr
              unique_ages[4]:'limegreen', # 2.5 Myr
              unique_ages[6]:'c', #3.5 Myr
              unique_ages[9]:'blue'} # 5 Myr

##############################
# replicate fig. 8 of gouliermis+ 2012
##############################

plt.figure(figsize=(6,8))

# plot PMS stars
plt.scatter(PMS_df['VminI'],
            PMS_df['Imag'],c='lightsteelblue',s=2,alpha=0.6)
# for larger point in legend
plt.scatter(0,0,c='lightsteelblue',label='PMS star')

# plot isochrones
#for i in range(len(unique_ages)):
for i in [0,1,2,4,6,9]:
    des_age = unique_ages[i]
    plt.plot(pisa_phot['555min814'][pisa_phot['log_age_yr']==des_age],
             pisa_phot['acswfc_f814w'][pisa_phot['log_age_yr']==des_age],
             c=fin_colors[des_age],label='%.1f'%(10**des_age/1e6)+' Myr')
# formatting
plt.legend(frameon=True,fontsize=9)
plt.xlabel('F555W - F814W [mag]')
plt.ylabel('F814W [mag]')
plt.xlim(-0.3,3.1)
plt.ylim(25.7,17.6)
plt.tight_layout()






