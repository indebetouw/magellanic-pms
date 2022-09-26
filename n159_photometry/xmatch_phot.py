
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import pandas as pd
import vaex as vx
import os
from astropy.table import Table
from astropy.wcs import wcs
import subprocess
import cmasher as cmr

photdir = '/Users/toneill/N159/photometry/'
region = "n159s"
workdir = photdir+region+'.phot0/'

filts = ["f555w", "f814w"]
#filts = ["f125w", "f160w"]

#f814_cols = pd.read_csv(photdir+'n159s.phot0/f814w/n159s_f814w_f814ref.columns',sep='\\',header=None)[0]


#os.chdir(workdir + region + "/")

c = {}
for filt in filts:
    if filt == "f125w" or filt == "f160w":
        #kind = "flt"
        #camera = "wfc3"
        #catfile = filt + "/" + region + "_" + filt + "_f160ref"
        catfile = photdir+'n159-w_'+filt+'_f160ref'
    else:
        # this is of course particular to this project that we did UVO with ACS and IR with WFC3
        kind = "flc"
        camera = "acs"
        catfile = workdir+filt + "/" + region + "_" + filt + "_f814ref"

    print(catfile)
    c[filt] = Table.read(catfile, format="ascii")
    print(len(c[filt]))
    print(c[filt][0:5])

# column definitions
mag = 'col16' # Instrumental VEGAMAG magnitude, ACS_F814W
dmag= 'col18' #18. Magnitude uncertainty, ACS_F814W
snr = 'col6' # 6. Signal-to-noise
shp = 'col7' #7. Object sharpness
rnd = 'col8' # 8. Object roundness
x   = 'col3' # 3. Object X position on reference image
y   = 'col4' #  4. Object Y position on reference image
otype = 'col11' #11. Object type (1=bright star, 2=faint, 3=elo...
crd = 'col10' # 10. Crowding

'''
dmag<0.1 (or SNR>10),
mag < 90 (it indicates nondetection with 99 or 100)
crowding<0.48 (not sure where 0.48 came from but that’s what PHAT settled on so we can quote them and move on:
sharp>-0.6
'''

if filts == ['f555w', 'f814w']:

    pass_555 = c['f555w'][((c['f555w'][snr] >= 10) & (c['f555w'][mag] < 90) & \
                           (c['f555w'][crd] <= 0.48) & (c['f555w'][shp] > -0.6))]
    pass_814 = c['f814w'][((c['f814w'][snr] >= 10) & (c['f814w'][mag] < 90) & \
                           (c['f814w'][crd] <= 0.48) & (c['f814w'][shp] > -0.6))]

    ref_head = fits.open('/Users/toneill/N159/photometry/ref_files_WCS/n159s_f814w_drc_sci.chip0.fits')[0].header
    ref_head['TARGNAME']

    # ref_head_814 = fits.open('/Users/toneill/N159/photometry/ref_files_WCS/f814w_drc_sci.chip0.fits')[0].header
    # ref_head_814['TARGNAME']

    ref_wcs = wcs.WCS(ref_head)
    for p in [pass_160, pass_125]:
        xpix = p[x]
        ypix = p[y]
        ra, de = ref_wcs.wcs_pix2world(xpix, ypix, 0)
        p["ra"] = ra
        p["dec"] = de

    from astropy.coordinates import SkyCoord
    from astropy import units as u

    c160 = SkyCoord(ra=pass_160['ra'] * u.deg, dec=pass_160['dec'] * u.deg)
    c125 = SkyCoord(ra=pass_125['ra'] * u.deg, dec=pass_125['dec'] * u.deg)
    max_sep = 0.2 * u.arcsec
    idx, d2d, d3d = c125.match_to_catalog_sky(c160)
    sep_constraint = d2d < max_sep
    print(np.sum(sep_constraint))
    c125_matches = pass_125[sep_constraint]
    c160_matches = pass_160[idx[sep_constraint]]
    c125_cat = c125_matches[['ra', 'dec', x, y, mag, dmag, snr, shp, rnd, otype, crd]].to_pandas()
    c125_cat.columns = [cname + '_125' for cname in
                        ['ra', 'dec', 'x', 'y', 'mag', 'dmag', 'snr', 'shp', 'rnd', 'otype', 'crd']]
    c160_cat = c160_matches[['ra', 'dec', x, y, mag, dmag, snr, shp, rnd, otype, crd]].to_pandas()
    c160_cat.columns = [cname + '_160' for cname in
                        ['ra', 'dec', 'x', 'y', 'mag', 'dmag', 'snr', 'shp', 'rnd', 'otype', 'crd']]
    cmatch = c160_cat.join(c125_cat)
    cmatch['125min160'] = cmatch['mag_125'] - cmatch['mag_160']

    cmatch.to_csv('n159-w_xmatch125.160_f160ref.csv', index=False)

    plt.figure()
    plt.scatter(cmatch['125min160'], cmatch['mag_160'], s=1, alpha=0.9)
    plt.gca().invert_yaxis()
    plt.xlabel('F125 - F160 [mag]')
    plt.ylabel('F125 [mag]')
    plt.colorbar()
    # plt.show()

    xs = cmatch['125min160']
    ys = cmatch['mag_160']

    # here, selected to make life easier by having "square" bins)
    xbins = np.arange(np.min(xs), np.max(xs), 0.1)
    ybins = np.linspace(np.min(ys), np.max(ys), len(xbins))

    # create fig & plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    hdict = hess_bin(filt1=xs, filt2=ys,
                     xbins=xbins, ybins=ybins,
                     ax=ax, fig=fig, cm='viridis')
    ax.set_xlabel('F125 - F160')
    ax.set_ylabel('F160')

if filts == ["f125w", "f160w"]:
    pass_160 = c['f160w'][((c['f160w'][snr] >=10) & (c['f160w'][mag] < 90) & \
            (c['f160w'][crd]<=0.48) & (c['f160w'][shp]>-0.6))]
    pass_125 = c['f125w'][((c['f125w'][snr] >=10) & (c['f125w'][mag] < 90) & \
            (c['f125w'][crd]<=0.48) & (c['f125w'][shp]>-0.6))]

    ref_head = fits.open('/Users/toneill/N159/photometry/ref_files_WCS/n159w_f160w_drz_sci.chip0.fits')[0].header
    ref_head['TARGNAME']

    #ref_head_814 = fits.open('/Users/toneill/N159/photometry/ref_files_WCS/f814w_drc_sci.chip0.fits')[0].header
    #ref_head_814['TARGNAME']

    ref_wcs = wcs.WCS(ref_head)
    for p in [pass_160,pass_125]:
        xpix=p[x]
        ypix=p[y]
        ra,de=ref_wcs.wcs_pix2world(xpix,ypix,0)
        p["ra"]=ra
        p["dec"]=de

    from astropy.coordinates import SkyCoord
    from astropy import units as u

    c160 = SkyCoord(ra=pass_160['ra']*u.deg,dec=pass_160['dec']*u.deg)
    c125 = SkyCoord(ra=pass_125['ra']*u.deg,dec=pass_125['dec']*u.deg)
    max_sep = 0.2 * u.arcsec
    idx, d2d, d3d = c125.match_to_catalog_sky(c160)
    sep_constraint = d2d < max_sep
    print(np.sum(sep_constraint))
    c125_matches = pass_125[sep_constraint]
    c160_matches = pass_160[idx[sep_constraint]]
    c125_cat = c125_matches[['ra','dec',x,y,mag,dmag,snr,shp,rnd,otype,crd]].to_pandas()
    c125_cat.columns = [cname+'_125' for cname in ['ra','dec','x','y','mag','dmag','snr','shp','rnd','otype','crd']]
    c160_cat = c160_matches[['ra','dec',x,y,mag,dmag,snr,shp,rnd,otype,crd]].to_pandas()
    c160_cat.columns = [cname+'_160' for cname in ['ra','dec','x','y','mag','dmag','snr','shp','rnd','otype','crd']]
    cmatch = c160_cat.join(c125_cat)
    cmatch['125min160'] = cmatch['mag_125'] - cmatch['mag_160']
    cmatch.to_csv('n159-w_xmatch125.160_f160ref.csv',index=False)


    ##########

    cmatch = pd.read_csv(photdir+'n159-w_xmatch125.160_f160ref.csv')

    ### make hess diagrams
    plt.figure()
    plt.scatter(cmatch['125min160'],cmatch['mag_160'],s=1,alpha=0.9)
    plt.gca().invert_yaxis()
    plt.xlabel('F125 - F160 [mag]')
    plt.ylabel('F125 [mag]')
    plt.colorbar()
    #plt.show()

    xs = cmatch['125min160']
    ys = cmatch['mag_160']
    xbins = np.arange(np.min(xs), np.max(xs), 0.1)
    ybins = np.linspace(np.min(ys), np.max(ys), len(xbins))

    # create fig & plot
    fig = plt.figure(figsize=(12,6))
    ax = fig.add_subplot(122)
    hdict = hess_bin(filt1=xs, filt2=ys,
                     xbins=xbins, ybins=ybins,
                     ax=ax, fig=fig,cm=cmr.ember)
    ax.set_xlabel('F125 - F160 [mag]',fontsize=14)
    ax.set_ylabel('F160 [mag]',fontsize=14)

    xs = cmatch['125min160']
    ys = cmatch['mag_125']
    xbins = np.arange(np.min(xs), np.max(xs), 0.1)
    ybins = np.linspace(np.min(ys), np.max(ys), len(xbins))

    ax2 = fig.add_subplot(121)
    hdict = hess_bin(filt1=xs, filt2=ys,
                     xbins=xbins, ybins=ybins,
                     ax=ax2, fig=fig,cm=cmr.ember)
    ax2.set_xlabel('F125 - F160 [mag]',fontsize=14)
    ax2.set_ylabel('F125 [mag]',fontsize=14)

    fig.tight_layout()


    ############ create kde
    fig = plt.figure()
    ax = fig.add_subplot(111)
    twoD_kde(xs,ys,ax=ax,fig=fig,discrete=False)
    ax.invert_yaxis()











n159s_f814w_drc_sci = fits.open('/Users/toneill/N159/photometry/ref_files_WCS/n159s_f814w_drc_sci.chip0.fits')[0].header
n159w_f814w_drc_sci = fits.open('/Users/toneill/N159/photometry/ref_files_WCS/n159w_f814w_drc_sci.chip0.fits')[0].header
n159w_f160w_drz_sci = fits.open('/Users/toneill/N159/photometry/ref_files_WCS/n159w_f160w_drz_sci.chip0.fits')[0].header
n159s_f160w_drz_sci = fits.open('/Users/toneill/N159/photometry/ref_files_WCS/n159s_f160w_drz_sci.chip0.fits')[0].header

n159s_f814w_drc_sci['TARGNAME']
n159w_f814w_drc_sci['TARGNAME']

n159w_f160w_drz_sci['TARGNAME']
n159s_f160w_drz_sci['TARGNAME']