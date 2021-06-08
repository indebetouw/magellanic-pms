# show CMD of small region

from astropy.table import Table
import os,pickle,sys
import numpy as np
import matplotlib.pyplot as pl
if not os.path.abspath("../code") in sys.path:
    sys.path.append(os.path.abspath("../code"))
from kNN_extinction_map import kNN_extinction
pl.ion()

#########
# autoset catalog path based on user
if os.environ['USER'] =='toneill':
    catalogdir = '/Users/toneill/Box/MC/HST/'
# can expand for others if desired
else:
    catalogdir="../../MCBox/HST/"


##########

# region of interest

# clump72
name0="clump72"
ra0=84.736
de0=-69.071
rad0=0.2/60

# R136 as used by VK
ra136 =84.67625
de136 =-69.10092
rad136=4./60

colname=['f555w','f775u']  
magname=['f555w']

colname=['f110w','f160w']  
magname=['f110w']

# TODO something for combining WCS and ACS in 775

try:
    n=len(fullcat)
except:
    n=0
if n<=0:
    fullcatname=catalogdir+"HTTP.2015_10_20.1.astro"
    if os.path.exists(fullcatname+".pkl"):
        fullcat=pickle.load(open(fullcatname+".pkl",'rb'))
    else:
        fullcat=Table.read(fullcatname,format="ascii")
        pickle.dump(fullcat,open(fullcatname+".pkl",'wb'))


RA=fullcat['Ra'].data
Dec=fullcat['Dec'].data

dx=np.absolute((RA-ra0)*np.cos(de0*np.pi/180))
dy=np.absolute(Dec-de0)
    
zz0=np.where((dx<rad0)*(dy<rad0))[0]
d2=dx[zz0]**2+dy[zz0]**2

z0=zz0[np.where( (d2<rad0**2)*
                 (fullcat["m_"+colname[0]][zz0]<30)*
                 (fullcat["m_"+colname[1]][zz0]<30)*
                 (fullcat["m_"+magname[0]][zz0]<30))[0]]

# R136
dx=np.absolute((RA-ra136)*np.cos(de136*np.pi/180))
dy=np.absolute(Dec-de136)
zz0=np.where((dx<rad136)*(dy<rad136))[0]
d2=dx[zz0]**2+dy[zz0]**2
z136=zz0[np.where( (d2<rad136**2)*
                   (fullcat["m_"+colname[0]][zz0]<30)*
                   (fullcat["m_"+colname[1]][zz0]<30)*
                   (fullcat["m_"+magname[0]][zz0]<30))[0]]

#pl.clf()
color136= ( fullcat["m_"+colname[0]] - fullcat["m_"+colname[1]] )[z136]
mag136  = fullcat["m_"+magname[0]][z136]

color0= ( fullcat["m_"+colname[0]] - fullcat["m_"+colname[1]] )[z0]
mag0  = fullcat["m_"+magname[0]][z0]

pl.figure()
pl.subplot(121)
pl.plot(color136,mag136,'k,')
pl.plot(color0,mag0,'.',color='green')

pl.xlim(-2,5)
pl.ylim(28,10)
pl.xlabel(colname[0].upper() + " - " + colname[1].upper())
pl.ylabel(magname[0].upper())

# median line as a function of mag
magsteps=10+np.arange(18)
trace=np.zeros(18)
for i in range(17):
    z=np.where( (mag136>=magsteps[i])*(mag136<magsteps[i+1])*(color136>(-2)*(color136<4)) )[0]
    trace[i]=np.median(color136[z])
z=np.where(trace>0)[0]
pl.plot(trace[z],magsteps[z]+0.5,'r')



#f=open(name0+"."+colname[0]+"-"+colname[1]+"_"+magname[0]+".all.reg","w")
#f.write("fk5\n")
#for i in range(len(z0)):
#    f.write('circle(%f,%f,0.5")\n'%(RA[z0[i]],Dec[z0[i]]))

#f.close()


#----------------------------------------
# now do the extinction correction

ums=Table.read(catalogdir+"Ksoll2018_HTTP_UMS_selection.csv",format="ascii")
URA = ums['Ra'].data
UDec = ums['Dec'].data
UAV = ums['A_v'].data

nnear=20
eps=10./3600           

# de Marchi 2016 extinction law
# https://academic.oup.com/mnras/article/455/4/4373/1264525
R_BV = [4.48, 3.03, 1.83, 1.22] # at V,I,J,H
label = ['555','775', '110', '160']

color0_dered=color0.copy()
color136_dered=color136.copy()
mag0_dered=mag0.copy()
mag136_dered=mag136.copy()

av0   = np.array([kNN_extinction(URA,UDec,UAV,eps,nnear,ri,di) for ri,di in zip(RA[z0],Dec[z0])])
av136 = np.array([kNN_extinction(URA,UDec,UAV,eps,nnear,ri,di) for ri,di in zip(RA[z136],Dec[z136])])
for l,rbv in zip(label,R_BV):
    if l in colname[0]:
        color0_dered = color0_dered - av0*rbv/R_BV[0]
        color136_dered = color136_dered - av136*rbv/R_BV[0]
    if l in colname[1]:
        color0_dered = color0_dered + av0*rbv/R_BV[0]
        color136_dered = color136_dered + av136*rbv/R_BV[0]
    if l in magname[0]:
        mag0_dered = mag0_dered - av0*rbv/R_BV[0]
        mag136_dered = mag136_dered - av136*rbv/R_BV[0]
    

# median line as a function of mag 
trace_dered=np.zeros(18)
for i in range(17):
    z=np.where( (mag136_dered>=magsteps[i])*(mag136_dered<magsteps[i+1])*(color136_dered>(-2)*(color136_dered<4)) )[0]
    trace_dered[i]=np.median(color136_dered[z])

        
pl.subplot(122)
pl.plot(color136_dered,mag136_dered,'k,')
# median line before color correction
z=np.where(trace>0)[0]
pl.plot(trace[z],magsteps[z]+0.5,'k',linestyle='dashed')

pl.plot(color0_dered,mag0_dered,'.',color='green')

pl.xlim(-2,5)
pl.ylim(28,10)
pl.xlabel(colname[0].upper() + " - " + colname[1].upper())
#pl.ylabel(magname[0].upper())





#----------------------------------------

# what about the identified UMS stars in this region?

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

pl.subplot(121)
#ucolor0= ( ums["m_"+colname[0]] - ums["m_"+colname[1]] )[uz0]
ucolor0= ( fread(ums,"m_"+colname[0]) - fread(ums,"m_"+colname[1]) )[uz0]
#umag0  = ums["m_"+magname[0]][uz0]
umag0  = fread(ums,"m_"+magname[0])[uz0]
pl.plot(ucolor0,umag0,'ro')

ucolor0_dered=ucolor0.copy()
umag0_dered=umag0.copy()

for l,rbv in zip(label,R_BV):
    if l in colname[0]:
        ucolor0_dered = ucolor0_dered - UAV[uz0]*rbv/R_BV[0]
    if l in colname[1]:
        ucolor0_dered = ucolor0_dered + UAV[uz0]*rbv/R_BV[0]
    if l in magname[0]:
        umag0_dered = umag0_dered - UAV[uz0]*rbv/R_BV[0]
        
pl.subplot(122)
pl.plot(ucolor0_dered,umag0_dered,'ro')
for i in range(len(ucolor0_dered)):
    pl.plot([ucolor0[i],ucolor0_dered[i]],[umag0[i],umag0_dered[i]],'r')
    pl.text(ucolor0_dered[i]-1,umag0_dered[i]-0.25,str(UAV[uz0[i]]),color='r')

    
#----------------------------------------

# what about the identified PMS stars in this region?

pms=Table.read(catalogdir+"Ksoll2018_HTTP_PMS_catalogue.csv",format="ascii")
PRA = pms['Ra'].data
PDec = pms['Dec'].data
PAV = pms['A_v'].data

dx=np.absolute((PRA-ra0)*np.cos(de0*np.pi/180))
dy=np.absolute(PDec-de0)
zz0=np.where((dx<rad0)*(dy<rad0))[0]
d2=dx[zz0]**2+dy[zz0]**2
pz0=zz0[np.where(d2<rad0**2)[0]]

#pcolor0= ( pms["m_"+colname[0]] - pms["m_"+colname[1]] )[pz0]
#pmag0  = pms["m_"+magname[0]][pz0]
pcolor0= ( fread(pms,"m_"+colname[0]) - fread(pms,"m_"+colname[1]) )[pz0]
pmag0  = fread(pms,"m_"+magname[0])[pz0]

pcolor0_dered=pcolor0.copy()
pmag0_dered=pmag0.copy()

for l,rbv in zip(label,R_BV):
    if l in colname[0]:
        pcolor0_dered = pcolor0_dered - PAV[pz0]*rbv/R_BV[0]
    if l in colname[1]:
        pcolor0_dered = pcolor0_dered + PAV[pz0]*rbv/R_BV[0]
    if l in magname[0]:
        pmag0_dered = pmag0_dered - PAV[pz0]*rbv/R_BV[0]

pl.plot(pcolor0_dered,pmag0_dered,'cx')

#f=open(name0+"."+colname[0]+"-"+colname[1]+"_"+magname[0]+".pms.reg","w")
#f.write("fk5\n")
#for i in range(len(pz0)):
#    f.write('circle(%f,%f,0.5") # text={%4.2f}\n'%(PRA[pz0[i]],PDec[pz0[i]],pms['p_mean'][pz0[i]]))
#f.close()

#pl.savefig(name0+".cmd."+colname[0]+"-"+colname[1]+"_"+magname[0]+".full.png")


# ----------------------------------------------
# zoom into red-corrected CMD
pl.figure()
#pl.clf()
pl.plot(color0_dered,mag0_dered,'.')

# median line of r136
z=np.where(trace_dered>0)[0]
pl.plot(trace_dered[z],magsteps[z]+0.5,'k',linestyle='dashed')
pl.plot(pcolor0_dered,pmag0_dered,'cx')
for i in range(len(pz0)):
    pl.text(pcolor0_dered[i],pmag0_dered[i],"%4.2f"%pms['p_mean'][pz0[i]],color='cyan')

pl.xlim(-1,3)
pl.ylim(28,18)
pl.xlabel(colname[0].upper() + " - " + colname[1].upper())
pl.ylabel(magname[0].upper())

# these are the indices when run with 555 and 755
special=[141,125,130,266,115]
# for the IR plot,
special=[294,151,163,145,135]
for i in special:
    pl.plot(color0_dered[i],mag0_dered[i],'ro')
    pl.text(color0_dered[i],mag0_dered[i],str(z0[i]),color='r')



