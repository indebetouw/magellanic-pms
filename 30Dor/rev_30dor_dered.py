#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 17:47:12 2021

@author: toneill
"""

from astropy.table import Table
import os,pickle,sys
import numpy as np
import matplotlib.pyplot as plt
if not os.path.abspath("../code") in sys.path:
    sys.path.append(os.path.abspath("../code"))
from kNN_extinction_map import kNN_regr_Av #kNN_extinction, 

# TODO something for combining WCS and ACS in 775

#########
# autoset catalog path based on user
if os.environ['USER'] =='toneill':
    catalogdir = '/Users/toneill/Box/MC/HST/'
# can expand for others if desired
else:
    catalogdir="../../MCBox/HST/"

########################## LOAD DATA
try:
    n=len(fullcat)
except:
    n=0
if n<=0:
    fullcatname=catalogdir+"HTTP.2015_10_20.1.astro"
    if os.path.exists(fullcatname+".pkl"):
        fullcat=pickle.load(open(fullcatname+".pkl",'rb')).to_pandas()
    else:
        fullcat=Table.read(fullcatname,format="ascii").to_pandas()
        pickle.dump(fullcat,open(fullcatname+".pkl",'wb'))

fullcat = fullcat.reset_index(drop=False)
fullcat = fullcat.rename(columns = {'index':'HTTP_index'})
fullcat['ksoll_ID'] = fullcat['HTTP_index'] + 1

########################### region of interest

# clump72 - REPLACED THIS TO MEAN ENTIRE SAMPLE
name0="clump72"
ra0=84.736
de0=-69.071
rad0=100000 # 0.2/60

# R136 as used by VK
ra136 =84.67625
de136 =-69.10092
rad136=4./60

colname=['f555w','f775u']  
magname=['f555w']
#colname=['f110w','f160w']  
#magname=['f110w']

RA=fullcat['Ra'].values
Dec=fullcat['Dec'].values

dx=np.absolute((RA-ra0)*np.cos(de0*np.pi/180))
dy=np.absolute(Dec-de0)
    
zz0=np.where((dx<rad0)*(dy<rad0))[0]
d2=dx[zz0]**2+dy[zz0]**2

z0=zz0[np.where( (d2<rad0**2)*
                 (fullcat["m_f555w"][zz0]<30)*
                 (fullcat["m_f775u"][zz0]<30)*
                 (fullcat['f_f555w']<=2)*
                 (fullcat['f_f775u']<=2))[0]]

color0= ( fullcat["m_"+colname[0]] - fullcat["m_"+colname[1]] )[z0].values
mag0  = fullcat["m_"+magname[0]][z0].values

trim_cat = fullcat.iloc[z0].reset_index(drop=False)

#----------------------------------------
# now do the extinction correction

color0_dered=color0.copy()
mag0_dered=mag0.copy()

ums=Table.read(catalogdir+"Ksoll2018_HTTP_UMS_selection.csv",format="ascii").to_pandas()
URA = ums['Ra'].values
UDec = ums['Dec'].values
UAV = ums['A_v'].values

# de Marchi 2016 extinction law
# https://academic.oup.com/mnras/article/455/4/4373/1264525
R_BV = [4.48, 3.03, 1.83, 1.22] # at V,I,J,H
#R_BV = [4.48, 3.74, 1.83, 1.22] # at V,R,J,H

label = ['f555w','f775u', 'f110w', 'f160w']

# weighting params for KNN from Ksoll 2018
nnear=20
eps=10./3600           

# separate dfs into nx2 arrays for KNN
# TO DO: improve how done for TCoords - takes forever since list
UCoords = [[URA[i],UDec[i]] for i in range(len(URA))]
TCoords = [[RA[z0][i],Dec[z0][i]] for i in range(len(RA[z0]))]
#start = timer()
av0 = kNN_regr_Av(UCoords,TCoords,UAV,eps,nnear,ncore=7)
#end = timer()
#print(end - start) # takes <1 second
#from timeit import default_timer as timer
#start = timer()
#av0_old   = np.array([kNN_extinction(URA,UDec,UAV,eps,nnear,ri,di) \
#                  for ri,di in zip(RA[z0],Dec[z0])])  
#end = timer()
#print(end - start) # takes 1 minute
#pickle.dump(av0,open("HTTP_NN_avs.pkl",'wb'))
#av0 = pickle.load(open("HTTP_NN_avs.pkl",'rb'))


trim_cat['AvNN'] = av0
trim_cat['m_f555w_dered'] = trim_cat['m_f555w'] - av0*R_BV[0]/R_BV[0]
trim_cat['m_f775u_dered'] = trim_cat['m_f775u'] - av0*R_BV[1]/R_BV[0]
trim_cat['m_f110w_dered'] = trim_cat['m_f110w'] - av0*R_BV[2]/R_BV[0]
trim_cat['m_f160w_dered'] = trim_cat['m_f160w'] - av0*R_BV[3]/R_BV[0]

#trim_cat.to_csv(catalogdir+'trim_HTTP.2015_10_20.1.csv')

#### PRESERVING THE BELOW FOR SAKE OF HAVING SCRIPT RUN FOR PLOTTING
# BUT LINE ABOVE IS WHAT'S USEFUL FOR BULK WORK
for l,rbv in zip(label,R_BV):
    print(l,rbv)
    #print('l')
    if l in colname[0]:
        #print(l)
        color0_dered = color0_dered - av0*rbv/R_BV[0]
    if l in colname[1]:
        #print(l)
        color0_dered = color0_dered + av0*rbv/R_BV[0]
    if l in magname[0]:
        #print(l)
        mag0_dered = mag0_dered - av0*rbv/R_BV[0]
    
#----------------------------------------

# now deredden Ksoll-identified UMS stars in this region

dx=np.absolute((URA-ra0)*np.cos(de0*np.pi/180))
dy=np.absolute(UDec-de0)
zz0=np.where((dx<rad0)*(dy<rad0))[0]
d2=dx[zz0]**2+dy[zz0]**2
uz0=zz0[np.where(d2<rad0**2)[0]]

# TODO deal with ACS+WF3 in full catalog
colname[1]=colname[1][:-1]+'w'

# get other filters with -10 for 'NA' mags:
def fread(table,colname):
    x=table[colname].data # S6 array
    z=np.where(x=='NA')[0]
    x[z]='0'
    return(np.float32(x))

ucolor0= ( ums["m_"+colname[0]] - ums["m_"+colname[1]] )[uz0]
#ucolor0= ( fread(ums,"m_"+colname[0]) - fread(ums,"m_"+colname[1]) )[uz0]
umag0  = ums["m_"+magname[0]][uz0]
#umag0  = fread(ums,"m_"+magname[0])[uz0]

ucolor0_dered=ucolor0.copy()
umag0_dered=umag0.copy()

label2 = ['f555w','f775w', 'f110w', 'f160w']


for l,rbv in zip(label2,R_BV):
    if l in colname[0]:
        ucolor0_dered = ucolor0_dered - UAV[uz0]*rbv/R_BV[0]
        print(l,rbv)
    if l in colname[1]:
        ucolor0_dered = ucolor0_dered + UAV[uz0]*rbv/R_BV[0]
        print(l,rbv)
    if l in magname[0]:
        umag0_dered = umag0_dered - UAV[uz0]*rbv/R_BV[0]
        
        
    
#----------------------------------------

#----------------------------------------

# what about the identified PMS stars in this region?

pms=Table.read(catalogdir+"Ksoll2018_HTTP_PMS_catalogue.csv",format="ascii").to_pandas()
PRA = pms['Ra'].values
PDec = pms['Dec'].values
PAV = pms['A_v'].values

dx=np.absolute((PRA-ra0)*np.cos(de0*np.pi/180))
dy=np.absolute(PDec-de0)
zz0=np.where((dx<rad0)*(dy<rad0))[0]
d2=dx[zz0]**2+dy[zz0]**2
pz0=zz0[np.where(d2<rad0**2)[0]]

colname2 = ['f555w', 'f775w']

pcolor0= ( pms["m_"+colname2[0]] - pms["m_"+colname2[1]] )[pz0]
pmag0  = pms["m_"+magname[0]][pz0]
#pcolor0= ( fread(pms,"m_"+colname[0]) - fread(pms,"m_"+colname[1]) )[pz0]
#pmag0  = fread(pms,"m_"+magname[0])[pz0]

pcolor0_dered=pcolor0.copy()
pmag0_dered=pmag0.copy()

for l,rbv in zip(label2,R_BV):
    if l in colname[0]:
        pcolor0_dered = pcolor0_dered - PAV[pz0]*rbv/R_BV[0]
    if l in colname[1]:
        pcolor0_dered = pcolor0_dered + PAV[pz0]*rbv/R_BV[0]
    if l in magname[0]:
        pmag0_dered = pmag0_dered - PAV[pz0]*rbv/R_BV[0]
        
fig,[ax1,ax2] = plt.subplots(1,2,sharex=True,sharey=True)
ax1.scatter(color0,mag0,s=0.05,alpha=0.2,color='k',label='HTTP')
ax1.scatter(ucolor0,umag0,c='r',s=0.2,alpha=0.2,label='UMS')
ax1.scatter(pcolor0,pmag0,s=0.2,alpha=0.2,c='c',label='PMS')
ax1.set_xlim(-3,4)
ax1.set_ylim(28,10)
ax1.set_xlabel(colname[0].upper() + " - " + colname[1].upper())
ax1.set_ylabel(magname[0].upper())

ax2.scatter(color0_dered,mag0_dered,c='k',s=0.05,alpha=0.3,label='HTTP')
ax2.scatter(ucolor0_dered,umag0_dered,c='r',s=0.2,alpha=0.2,label='UMS')
ax2.scatter(pcolor0_dered,pmag0_dered,s=0.2,alpha=0.2,c='c',label='PMS')
ax2.set_xlabel('Dereddened '+colname[0].upper() + " - " + colname[1].upper())
ax2.set_ylabel('Dereddened '+magname[0].upper())

fig.tight_layout()      































