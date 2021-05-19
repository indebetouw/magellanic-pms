# find UMS stars at different CO intensity levels
# no trend is evident

from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
import matplotlib.pyplot as pl
pl.ion()

cofile="ALMA/30Dor_feather_mosaic83_12CO_7meter.mom0.fits"

cohdu=fits.open(cofile)[0]

ums=Table.read("Ksoll2018_HTTP_UMS_selection.csv",format="ascii")
URA = ums['Ra'].data
UDec = ums['Dec'].data
UAV = ums['A_v'].data

# get UMS stars locations in pixels of CO image
ux,uy = WCS(cohdu.header).wcs_world2pix(URA,UDec,0)

comax=np.nanmax(cohdu.data)
nlev=20

AVmax =np.zeros(nlev)+np.nan
AVmean=np.zeros(nlev)+np.nan
AVrms =np.zeros(nlev)+np.nan

levs=np.arange(nlev+1)*comax/(nlev)
# get contours as paths
cs = pl.contour(cohdu.data,levs,linewidths=1,colors='k')

import matplotlib.patches as patches

selected=[[]]*(nlev+1)

for ilev in range(nlev+1):
    verts = cs.collections[ilev].get_paths()
    usel=[]
    for i in range(len(verts)):
        if ilev==5:
            pl.gca().add_patch(patches.PathPatch(verts[i], facecolor='none', ec='yellow', lw=1, zorder=50))
        newu=np.where(verts[i].contains_points(np.array([ux,uy]).T))[0]
        if len(usel)==0:
            usel=newu
        elif len(newu)>0:
            usel=np.concatenate([usel,newu])
    if ilev==5:
        pl.plot(ux[usel],uy[usel],'rx')
    usel.sort()
    selected[ilev]=usel

for ilev in range(nlev):
    usel=np.setdiff1d(selected[ilev],selected[ilev+1])
    if len(usel)>0:
        print(levs[ilev],UAV[usel])
        AVmax[ilev] =np.nanmax(UAV[usel])
        AVmean[ilev]=np.nanmean(UAV[usel])
        AVrms[ilev]=np.nanstd(UAV[usel])

    
pl.clf()
# pl.plot(AVmean,levs[:-1],'o')
# pl.plot(AVmax,levs[:-1],'o')
pl.plot(levs[:-1],AVmax,'x')
pl.errorbar(levs[:-1],AVmean,yerr=AVrms,fmt='o')

pl.xlabel("CO integrated intensity")
pl.ylabel("UMS A_V")
pl.savefig("co_avstars.png")
