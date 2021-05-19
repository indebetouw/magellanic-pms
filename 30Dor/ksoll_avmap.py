from astropy.table import Table
import os, sys
from astropy.io import fits
from astropy.wcs import WCS

from astropy import coordinates
from astropy import units as u
import scipy.ndimage
import numpy as np

import matplotlib.pyplot as pl
pl.ion()

if not os.path.abspath("../code") in sys.path:
    sys.path.append(os.path.abspath("../code"))

from kNN_extinction_map import kNN_extinction_map

# read Ksoll UMS list
# "ID","m_f275w","e_f275w","q_f275w","f_f275w","m_f336w","e_f336w","q_f336w","f_f336w","m_f555w","e_f555w","q_f555w","f_f555w","m_f658n","e_f658n","q_f658n","f_f658n","m_f775w","e_f775w","q_f775w","f_f775w","m_f110w","e_f110w","q_f110w","f_f110w","m_f160w","e_f160w","q_f160w","f_f160w","x","y","x_new","y_new","Ra","Dec","A_v","delta_Av"

ums=Table.read("Ksoll2018_HTTP_UMS_selection.csv",format="ascii")
URA = ums['Ra'].data
UDec = ums['Dec'].data
UAV = ums['A_v'].data

# map range:
# ALMA 84.564~84.824  -69.145~-69.012
xra=[84.56,84.83]
yra=[-69.15,-69.01]
prefix="almaregion"

# UMS  84.136~84.988  -68.997~-69.261
xra=[84.135,85.0]
yra=[-69.265,-68.99]
prefix="full"

nnear=20
cell=5.
epsilon=10.
avroot="30dor_avmap_"+prefix+("_N%i_e%03isec_d%02isec"%(nnear,epsilon,cell))

# load or create the AV map:
if os.path.exists(avroot+".fits"):
    avhdu=fits.open(avroot+".fits")[0]
    avmap=avhdu.data
    avwcs=WCS(avhdu.header)
else:
    avmap,avwcs = kNN_extinction_map(URA,UDec,UAV,xra,yra, avroot,
                                     nnear=nnear, epsilon=epsilon, cell=cell)



# =====================================================
# project AV map to CO and display



coroot="30Dor_feather_mosaic83_12CO_7meter.mom0"


cohdu=fits.open("ALMA/"+coroot+".fits")[0]
avhdu=fits.open(avroot+".fits")[0]
hdr1=avhdu.header
hdr2=cohdu.header


# map AVmap to CO
wcs1 = WCS(hdr1)
wcs2 = WCS(hdr2)
# output shape
outshape=[hdr2['naxis2'],hdr2['naxis1']]
yy2, xx2 = np.indices(outshape)
# output world coords
lon2, lat2 = wcs2.wcs_pix2world(xx2,yy2,0)
# assumes same CTYPE ***
xx1, yy1 = wcs1.wcs_world2pix(lon2,lat2,0)
grid = np.array([yy1.reshape(outshape),xx1.reshape(outshape)])

av_mapped = scipy.ndimage.map_coordinates(avhdu.data, grid, order=1, mode='constant', cval=np.nan)


#-------------------
pl.clf()

ax=pl.subplot(111,projection = wcs2)
foo=ax.imshow(av_mapped,cmap="gray",interpolation="nearest")
ax.set_xlabel("Right Ascension (J2000)")
ax.set_ylabel("Declination")
pl.colorbar(ax=ax,label="A_V from UMS",mappable=foo)
ax.contour(cohdu.data)

pl.savefig(avroot+"_"+coroot+".png")


#pl.clf()
#z=np.where(cohdu.data > 0.5)
#pl.plot(av_mapped[z],cohdu.data[z],',')




