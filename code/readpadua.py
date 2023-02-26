root="Z0.004Y0.256"
root="Z0.02Y0.284"

from astropy.table import Table
import matplotlib.pyplot as pl
import numpy as np

# not clear what the HB are.
#hb=Table.read(root+"/Z0.004Y0.256OUTA1.74_F7_M0.750.HB.DAT",format="ascii")
#reg=Table.read(root+"/Z0.004Y0.256OUTA1.74_F7_M000.750.DAT",format="ascii")

#reg=Table.read(root+"/Z0.004Y0.256OUTA1.74_F7_M005.000.DAT",format="ascii")
#pl.plot(reg['AGE'],reg['LOG_TE'])
#z=pl.where(reg['PHASE']<7)[0]
#pl.plot(reg['AGE'][z],reg['LOG_TE'][z])
#z=pl.where(reg['PHASE']<4)[0]
#pl.plot(reg['AGE'][z],reg['LOG_TE'][z])

from glob import glob
f0=glob(root+"/"+root+"OUTA1.74_F7_M???.???.DAT")

ff=[]
m=[]
for i in range(len(f0)):
    thism=float(f0[i].split("M")[1][0:7])
    if thism>=1:
        ff.append(f0[i])
        m.append(thism)

nm=len(ff)
# 1Msun: ~1500pts between 1 and 6.5e9yr
# 100  : 600 pts between 1 and 3e3 yr
lage=np.arange(1500)/1500*9.8
nt=len(lage)

# index by m0 and age
mass=np.zeros([nm,nt])
logl=np.zeros([nm,nt])
logt=np.zeros([nm,nt])
logr=np.zeros([nm,nt])
phase=np.zeros([nm,nt])

pl.clf()
ff.sort()

for i in range(nm):
    print(ff[i])
    t=Table.read(ff[i],format="ascii")
    logage=np.log10(t['AGE'].data)
    for j in range(nt):
        dt=np.absolute(logage-lage[j])
        z=np.where(dt==dt.min())[0][0]

#  1 PMS_BEG  Track begins here (Pre main sequence)
#  2 PMS_MIN  
#  3 PMS_END    PMS is near to end
#  4 NEAR_ZAM   This point is very near the ZAMS
#  5 MS_BEG     H burning fully active
#  6 POINT_B    Almost end of the H burning. Small contraction phase begins here for interm. & massive stars   
#  7 POINT_C    Small contraction ends here and star move toward RG
#  8 RG_BASE    RG base
#  9 RG_BMP1   RGB bump in Low Mass Stars (marked also for other masses)
# 10 RG_BMP2   RGB bump end in Low Mass Stars (marked also for other masses)
# 11 RG_TIP   Helium Flash or beginning of HELIUM Burning in intermediate and massive stars
# 12 Loop_A   Base of Red He burning, before possible loop
# 13 Loop_B   Bluest point of the loop (He burning
# 14 Loop_C   central He = 0  almost 
# 15 TPAGB begins or c_burning begins (massive stars, generally if LC > 2.9)

        
        if (t['PHASE'][z]<=4)*(t['PHASE'][z]>0):  # ditch the high-mass cooling as burning, to avoid loops in HR and weirdness when interpolating.
#        if (t['PHASE'][z]>0):  
            mass[i,j]=t['MASS'][z]
            logl[i,j]=t['LOG_L'][z]
            logt[i,j]=t['LOG_TE'][z]
            logr[i,j]=t['LOG_R'][z]
            phase[i,j]=t['PHASE'][z]
    z=np.where(mass[i]>0)[0]
    pl.plot(logt[i][z],logl[i][z])

    
pl.xlim(pl.xlim()[::-1])

import pickle
pickle.dump([lage,mass,logl,logt,logr,phase],open(root+".pkl","wb"))




#j=60
#pl.clf()
#pl.plot(logt[j],logl[j],'k')
#for i in range(9):
#    z=pl.where((phase[j]>i)*(phase[j]<=(i+1)))[0]
#    pl.plot(logt[j][z],logl[j][z],'.',label=str(i))
#
#pl.legend(loc="best",prop={"size":8})
#pl.xlim(pl.xlim()[::-1])
