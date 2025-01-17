import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from matplotlib.patches import Circle, Rectangle, Arrow, FancyBboxPatch
from skimage import measure
from reproject import reproject_interp
#from hcongrid import hcongrid
#import img_scale
import copy
import numpy as np
from astropy.io import fits
from astropy import units as u
#import aplpy

import pandas as pd
savephotdir = '/Users/toneill/N159/photometry/reduced/'

fig = plt.figure()
img = aplpy.FITSFigure(refdir+'f814_n159.fits',figure=fig)
img.show_grayscale(stretch='arcsinh')
img.add_scalebar(length=0.05729*u.deg)
img.scalebar.set_color('red')
img.scalebar.set_label('50 pc')
img.scalebar.set_corner('top right')
fig.show()

duse = 10**(1+18.48/5)
scale1_len_arcsec = 206265. * 1 / duse # 1 pc in arcsec # pc /arcsec
scale1_len = (scale1_len_arcsec*u.arcsec).to(u.pixel,
                    u.pixel_scale(ne_8[0].header['CD1_2']*u.deg/u.pixel))
print(scale1_len)



def summ_obs(filt):
    head = filt[0].header
    #print(f"exptime: {head['EXPTIME']:.0f}s")
    arcs_ppix = (np.abs(head['CDELT1'])*u.deg/u.pixel).to(u.arcsec/u.pixel)
    npix1 = head['CRPIX1']*u.pix*2
    npix2 = head['CRPIX2']*u.pix*2
    fov1 = npix1*arcs_ppix
    fov2 = npix2*arcs_ppix
    print(f'{fov1.to(u.arcmin):.1f} x {fov2.to(u.arcmin):.1f}')
    print(f'{fov1  / (scale1_len_arcsec*(u.arcsec/u.pc)):.0f}  x {fov2  / (scale1_len_arcsec*(u.arcsec/u.pc)):.0f} ')

summ_obs(n_8)

summ_obs(ne_5)
summ_obs(ne_8)
summ_obs(ne_12)
summ_obs(ne_16)

summ_obs(nw_5)
summ_obs(nw_8)
summ_obs(nw_12)
summ_obs(nw_16)

summ_obs(ns_5)
summ_obs(ns_8)
summ_obs(ns_12)
summ_obs(ns_16)

summ_obs(off_5)
summ_obs(off_8)
summ_obs(off_12)
summ_obs(off_16)


def total_area(filts):
    sqarea = 0
    for filt in filts:
        head = filt[0].header
        #print(f"exptime: {head['EXPTIME']:.0f}s")
        arcs_ppix = (np.abs(head['CD1_2'])*u.deg/u.pixel).to(u.arcsec/u.pixel)
        npix1 = head['CRPIX1']*u.pix*2
        npix2 = head['CRPIX2']*u.pix*2
        fov1 = npix1*arcs_ppix
        fov2 = npix2*arcs_ppix
        #print(f'{fov1.to(u.arcmin):.1f} x {fov2.to(u.arcmin):.1f}')
        #print(f'{fov1  / (scale1_len_arcsec*(u.arcsec/u.pc)):.0f}  x {fov2  / (scale1_len_arcsec*(u.arcsec/u.pc)):.0f} ')
        sqarea += (fov1  / (scale1_len_arcsec*(u.arcsec/u.pc))*fov2  / (scale1_len_arcsec*(u.arcsec/u.pc)))
    print(f'{sqarea:.0f} ')
    print(f' equiv ({np.sqrt(sqarea.value):0f} pc)^2')

total_area([ne_8,nw_8,ns_8,off_8])

total_area([ne_16,nw_16,ns_16,off_16])

#############
photdir = '/Users/toneill/N159/photometry/'
#refdir = photdir + 'ref_files_WCS/'
refdir = '/Users/toneill/N159/hst_mosaics/'



n_8 = fits.open(refdir+'f814_n159.fits')

img = np.sqrt(n_8[0].data)

fig = plt.figure(figsize=(10,10))
ax1 = fig.add_subplot(1,1,1,projection=WCS(n_8[0].header))
ax1.imshow(img,origin='lower',cmap='Greys_r',vmax=np.nanquantile(img,0.99))
ax1.set_xlabel('RA')
ax1.set_ylabel('Dec')
fig.tight_layout()
plt.savefig('/Users/toneill/N159/plots/n159_mosaic.pdf',dpi=300)


ne_8 = fits.open(refdir + 'f814_n159e.fits')
nw_8 = fits.open(refdir + 'f814_n159w.fits')
ns_8 = fits.open(refdir + 'f814_n159s.fits')
off_8 = fits.open(refdir + 'f814_off.fits')

ne_16 = fits.open(refdir + 'f160_n159e.fits')
nw_16 = fits.open(refdir + 'f160_n159w.fits')
ns_16 = fits.open(refdir + 'f160_n159s.fits')
off_16 = fits.open(refdir + 'f160_off.fits')

ne_5 = fits.open(refdir + 'f555_n159e.fits')
nw_5 = fits.open(refdir + 'f555_n159w.fits')
ns_5 = fits.open(refdir + 'f555_n159s.fits')
off_5 = fits.open(refdir + 'f555_off.fits')

ne_12 = fits.open(refdir + 'f125_n159e.fits')
nw_12 = fits.open(refdir + 'f125_n159w.fits')
ns_12 = fits.open(refdir + 'f125_n159s.fits')
off_12 = fits.open(refdir + 'f125_off.fits')

con_ne_16 = hcongrid(ne_16[0].data,ne_16[0].header,
                 ne_8[0].header,preserve_bad_pixels=False)
con_ne_5 = hcongrid(ne_5[0].data,ne_5[0].header,
                 ne_8[0].header,preserve_bad_pixels=False)
con_ne_12 = hcongrid(ne_12[0].data,ne_12[0].header,
                 ne_8[0].header,preserve_bad_pixels=False)

con_nw_16 = hcongrid(nw_16[0].data,nw_16[0].header,
                 nw_8[0].header,preserve_bad_pixels=False)

con_ns_16 = hcongrid(ns_16[0].data,ns_16[0].header,
                 ns_8[0].header,preserve_bad_pixels=False)

#con_off_16 = hcongrid(off_16[0].data,off_16[0].header,
#                 off_8[0].header,preserve_bad_pixels=False)



n_8scale = asinh(n_8[0].data,scale_min=min_8,scale_max=max_8)

plt.figure()
plt.imshow(n_8scale,origin='lower')

##########################################

ord_16 = 0
ord_8 = 2
ord_5 = 1
#ord_12 = 2

max_16 = 5
max_8 = 1
min_8 = 0.00
min_16 = 0.00

max_5 = 0.5
min_5 = 0

max_12 = 5
min_12 = 0

##########################################

ne_8scale = asinh(ne_8[0].data,scale_min=min_8,scale_max=max_8)
ne_16scale = asinh(con_ne_16,scale_min=min_16,scale_max=max_16)
ne_5scale = asinh(con_ne_5,scale_min=min_5,scale_max=max_5)
ne_12scale = asinh(con_ne_12,scale_min=min_12,scale_max=max_12)

comb_16_12 = ne_12scale+ne_16scale
comb_16_12 = (comb_16_12)/2

ne_final = np.zeros((ne_8[0].data.shape[0],ne_8[0].data.shape[1],3),dtype=float)
ne_final[:,:,ord_16] = comb_16_12
#ne_final[:,:,ord_12] = ne_12scale
ne_final[:,:,ord_8] = ne_8scale
ne_final[:,:,ord_5] = ne_5scale

plt.figure()
plt.imshow(ne_final,origin='lower')

#ne_partial = copy.deepcopy(ne_final)
#ne_partial[:,:,ord_16] = np.zeros(np.shape(ne_16scale))


####################################

nw_8scale = asinh(nw_8[0].data,scale_min=min_8,scale_max=max_8)
nw_16scale = asinh(con_nw_16,scale_min=min_16,scale_max=max_16)

nw_final = np.zeros((nw_8[0].data.shape[0],nw_8[0].data.shape[1],3),dtype=float)
nw_final[:,:,ord_16] = nw_16scale
nw_final[:,:,ord_8] = nw_8scale
#nw_final[:,:,2] = nw_8scale

nw_partial = copy.deepcopy(nw_final)
nw_partial[:,:,ord_16] = np.zeros(np.shape(nw_16scale))

####################################

ns_8scale = asinh(ns_8[0].data,scale_min=min_8,scale_max=max_8)
ns_16scale = asinh(con_ns_16,scale_min=min_16,scale_max=max_16)

ns_final = np.zeros((ns_8[0].data.shape[0],ns_8[0].data.shape[1],3),dtype=float)
ns_final[:,:,ord_16] = ns_16scale
ns_final[:,:,ord_8] = ns_8scale
#ns_final[:,:,2] = ns_8scale

ns_partial = copy.deepcopy(ns_final)
ns_partial[:,:,ord_16] = np.zeros(np.shape(ns_16scale))
#####################################

####################################

'''off_8scale = asinh(off_8[0].data,scale_min=min_8,scale_max=max_8)
off_16scale = asinh(con_off_16,scale_min=min_16,scale_max=max_16)

off_final = np.zeros((off_8[0].data.shape[0],off_8[0].data.shape[1],3),dtype=float)
off_final[:,:,ord_16] = off_16scale
off_final[:,:,ord_8] = off_8scale
off_final[:,:,2] = off_8scale'''

##################################################################################

#fig = plt.figure(figsize=(9,9))
#ax1 = fig.add_subplot(2,4,(1,2), projection=WCS(nw_8[0].header))
#ax2 = fig.add_subplot(2,4,(3,4), projection=WCS(ne_8[0].header))
#ax3 = fig.add_subplot(2,4,(6,7), projection=WCS(ns_8[0].header))

'''ax1 = fig.add_subplot(2,2,1, projection=WCS(nw_8[0].header))
ax2 = fig.add_subplot(2,2,2, projection=WCS(ne_8[0].header))
ax3 = fig.add_subplot(2,2,4, projection=WCS(ns_8[0].header))
ax4 = fig.add_subplot(2,2,3, projection=WCS(off_8[0].header))'''

fig = plt.figure(figsize=(12,5))

ax1 = fig.add_subplot(1,3,2, projection=WCS(nw_8[0].header))
ax2 = fig.add_subplot(1,3,1, projection=WCS(ne_8[0].header))
ax3 = fig.add_subplot(1,3,3, projection=WCS(ns_8[0].header))

ax1.tick_params(direction='in', which='both', labelsize=0,ticksize=0)
ax2.tick_params(direction='in', which='both', labelsize=0,ticksize=0)
ax3.tick_params(direction='in', which='both', labelsize=0,ticksize=0)
for ax in [ax1,ax2,ax3]:#,ax4]:
    ax.set_xlabel(' a',fontsize=0)
    ax.set_ylabel('a ',fontsize=0)
ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax3.set_aspect('equal')

ax1.set_title('N159W',pad=20)
ax2.set_title('N159E')
ax3.set_title('N159S',pad=20)
#ax4.set_title('Off')

ax1.imshow(nw_final)
ax2.imshow(ne_final)
ax3.imshow(ns_final)
#ax4.imshow(off_final)
fig.tight_layout()

#plt.savefig('n159.png',dpi=300)
#fig.subplots_adjust(wspace=0.05,hspace=0.1,left=0.0,right=1,bottom=0.01)

##########################################################

fig = plt.figure(figsize=(12,5))

ax1 = fig.add_subplot(1,3,1, projection=WCS(ne_8[0].header))
ax2 = fig.add_subplot(1,3,2, projection=WCS(nw_8[0].header))
ax3 = fig.add_subplot(1,3,3, projection=WCS(ns_8[0].header))

ax1.imshow(ne_8scale.T, cmap=cmr.freeze)
ax2.imshow(nw_8scale,   cmap=cmr.freeze)
ax3.imshow(ns_8scale, cmap=cmr.freeze)

ax1.scatter(full['ra_814'][pms_inds], full['dec_814'][pms_inds], c=full_prob[:, 1][pms_inds],
           cmap=cmap_use, s=0.5, vmin=0, vmax=1, alpha=0.5, transform=ax1.get_transform('fk5'), zorder=2)



fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(111, projection=wcs.WCS(alma_co[0].header))

ax.imshow(ne_8scale, transform=ax.get_transform(wcs.WCS(hst_n159e[0].header)),cmap=cmr.freeze)#, cmap='Greys_r')

ax.imshow(nw_8scale, transform=ax.get_transform(wcs.WCS(hst_n159w[0].header)), alpha=1, zorder=0,
          cmap=cmr.freeze)
ax.imshow(ns_8scale, transform=ax.get_transform(wcs.WCS(hst_n159s[0].header)),cmap=cmr.freeze)


ax.scatter(full['ra_814'][pms_inds], full['dec_814'][pms_inds], c=full_prob[:, 1][pms_inds],
           cmap=cmap_use, s=2, vmin=0, vmax=1, alpha=0.5, transform=ax.get_transform('fk5'), zorder=2)

ax.imshow(hst_n159s[0].data, transform=ax.get_transform(wcs.WCS(hst_n159s[0].header)), vmax=0.5)
ax.contour(alma_co[0].data, levels=[5], colors=['r'], zorder=3)

##################################################################################

fig = plt.figure(figsize=(12,9))
ax1 = fig.add_subplot(2,3,1, projection=WCS(nw_8[0].header))
ax2 = fig.add_subplot(2,3,2, projection=WCS(ne_8[0].header))
ax3 = fig.add_subplot(2,3,3, projection=WCS(ns_8[0].header))

ax4 = fig.add_subplot(2,3,4, projection=WCS(nw_8[0].header))
ax5 = fig.add_subplot(2,3,5, projection=WCS(ne_8[0].header))
ax6 = fig.add_subplot(2,3,6, projection=WCS(ns_8[0].header))

'''ax1.set_title('N159W')
ax2.set_title('N159E')
ax3.set_title('N159S')

for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
    ax.set_xlabel(' ',fontsize=0)
    ax.set_ylabel(' ',fontsize=0)

ax1.imshow(nw_final)
ax2.imshow(ne_final)
ax3.imshow(ns_final)'''

pprops = {'s':0.01,'facecolor':'None','edgecolor':'darkorange','alpha':0.9,'marker':'o'}
ax4.imshow(nw_partial)
ax4.scatter(w_vi['ra_814'],w_vi['dec_814'],transform=ax4.get_transform('fk5'),**pprops)
ax5.imshow(ne_partial)
ax6.imshow(ns_partial)

fig.subplots_adjust(wspace=0.05)















