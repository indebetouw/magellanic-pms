
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

photdir = '/Users/toneill/N159/photometry/new/'
region = "n159-e"
region_shortname = 'n159e'
#workdir = photdir+region+'/'+'n159-e-newoptical/'##.phot0/'
workdir = '/Users/toneill/N159/photometry/new/n159-e/n159-e-newoptical/'

filts = ["f555w", "f814w"]
#filts = ["f125w", "f160w"]
#os.chdir(workdir + region + "/")

c = {}
for filt in filts:
    if filt == "f125w" or filt == "f160w":
        #kind = "flt"
        #camera = "wfc3"
        #catfile = filt + "/" + region + "_" + filt + "_f160ref"
        catdir = photdir+region+'/'+filt+'/'
        catfile = catdir +region+filt+ '_f160ref'
    else:
        catdir = photdir+region+'/'+filt+'/'
        catfile = catdir +region+'_'+filt+ '_f814ref'

    #print(catfile)
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
sky='col13'
cts='col12'

if filts == ['f555w', 'f814w']:
    os.chdir(workdir)

    #ref_head = fits.open('/Users/toneill/N159/photometry/ref_files_WCS/'+region_shortname+'_f814w_drc_sci.chip0.fits')[0].header
    reffile = filt + "/" + subprocess.getoutput( 'grep img_file ' + filt + '/dolparms_' + filt + ".multi.txt").split()[-1] + ".fits"
    ref_head = fits.open(reffile)[0].header
    # saturated detections:
    cat = c['f555w']
    satur_555 = np.zeros(len(cat))
    for i in range(len(cat)):
        print(i/len(cat)*100)
        if cat[mag][i] > 90:
            if (cat['col37'][i] > 7 and cat['col38'][i] > 100 and  cat['col40'][i] / cat['col41'][i] > 10) or \
                    (cat['col50'][i] > 7 and cat['col51'][i] > 100 and cat['col53'][i] / cat['col54'][i] > 10):
                satur_555[i] += 1
    cat['satur'] = satur_555
    cat = c['f814w']
    satur_814 = np.zeros(len(cat))
    for i in range(len(cat)):
        print(i/len(cat)*100)
        if cat[mag][i] > 90:
            if (cat['col37'][i] > 7 and cat['col38'][i] > 100 and  cat['col40'][i] / cat['col41'][i] > 10) or \
                    (cat['col50'][i] > 7 and cat['col51'][i] > 100 and cat['col53'][i] / cat['col54'][i] > 10):
                satur_814[i] += 1
    cat['satur'] = satur_814

    from astropy.coordinates import SkyCoord
    from astropy import units as u

    detec_satur = False
    if detec_satur:
        ref_wcs = wcs.WCS(ref_head)
        for p in [c['f555w'], c['f814w']]:
            xpix = p[x]
            ypix = p[y]
            ra, de = ref_wcs.wcs_pix2world(xpix, ypix, 0)
            p["ra"] = ra
            p["dec"] = de
        c5_sat = c['f555w']#[c['f555w']['satur'] > 0]
        c8_sat = c['f814w']#[c['f814w']['satur'] > 0]
        c555 = SkyCoord(ra=c5_sat['ra'] * u.deg, dec=c5_sat['dec'] * u.deg)
        c814 = SkyCoord(ra=c8_sat['ra'] * u.deg, dec=c8_sat['dec'] * u.deg)
        max_sep = 0.1 * u.arcsec
        idx, d2d, d3d = c814.match_to_catalog_sky(c555)
        sep_constraint = d2d < max_sep
        print(np.sum(sep_constraint))
        satur_c814_matches = c8_sat[sep_constraint]
        satur_c555_matches = c5_sat[idx[sep_constraint]]
        satur_c814_cat = satur_c814_matches[['ra', 'dec', x, y, mag, dmag, snr, shp, rnd, otype, crd,'satur']].to_pandas()
        satur_c814_cat.columns = [cname + '_814' for cname in
                            ['ra', 'dec', 'x', 'y', 'mag', 'dmag', 'snr', 'shp', 'rnd', 'otype', 'crd','satur']]
        satur_c555_cat = satur_c555_matches[['ra', 'dec', x, y, mag, dmag, snr, shp, rnd, otype, crd,'satur']].to_pandas()
        satur_c555_cat.columns = [cname + '_555' for cname in
                            ['ra', 'dec', 'x', 'y', 'mag', 'dmag', 'snr', 'shp', 'rnd', 'otype', 'crd','satur']]
        satur_cmatch_vi = satur_c555_cat.join(satur_c814_cat)
        satur_cmatch_vi['555min814'] = satur_cmatch_vi['mag_555'] - satur_cmatch_vi['mag_814']
        sat_either_bool = (satur_cmatch_vi['satur_555'] + satur_cmatch_vi['satur_814']) > 0
        satur_cmatch_vi = satur_cmatch_vi[sat_either_bool]
        satur_cmatch_vi.to_csv(photdir+region+'_xmatch_0.1sec.555.814_f814ref_satur.csv', index=False)



    ctssky_555 = [(c['f555w'][cts][i] / c['f555w'][sky][i]) > \
                        min(4,c['f555w'][crd][i] * 10) for i in range(len(c['f555w']))]
    ctssky_814 = [(c['f814w'][cts][i] / c['f814w'][sky][i]) > \
                        min(4,c['f814w'][crd][i] * 10) for i in range(len(c['f814w']))]

    pass_555 = c['f555w'][ (c['f555w'][shp] < 0.27)   & (c['f555w'][dmag] < 0.1 ) &
                           (c['f555w'][crd] < 1) & ctssky_555 & (c['f555w'][mag] < 25)]
    pass_814 = c['f814w'][ (c['f814w'][shp] < 0.27)   & (c['f814w'][dmag] < 0.1 ) &
                           (c['f814w'][crd] < 1) & ctssky_814 & (c['f814w'][mag] < 25)]

    ref_wcs = wcs.WCS(ref_head)
    for i in range(2):
        p = [pass_555, pass_814][i]
        xpix = p[x]
        ypix = p[y]
        ra, de = ref_wcs.wcs_pix2world(xpix, ypix, 0)
        p["ra"] = ra
        p["dec"] = de

   '''   ref_wcs = wcs.WCS(ref_head)
        for p in [c['f555w'], c['f814w']]:
            xpix = p[x]
            ypix = p[y]
            ra, de = ref_wcs.wcs_pix2world(xpix, ypix, 0)
            p["ra"] = ra
            p["dec"] = de'''

    from astropy.coordinates import SkyCoord
    from astropy import units as u


    pd_555 = pass_555.to_pandas()#.to_csv(photdir+region+'_555only_f814ref.csv', index=False)
    pd_814 = pass_814.to_pandas()#.to_csv(photdir+region+'_814only_f814ref.csv', index=False)


    '''zsky5 =  c['f555w'][sky]== 0
    zsky8 = c['f814w'][sky]==0

    plt.figure()
    plt.scatter(c['f555w']['ra'][zsky5],c['f555w']['dec'][zsky5],s=1,marker='x',c='r')#c['f555w'][sky])
    plt.title('f555w sky = 0, n159e')

    plt.figure()
    plt.scatter(c['f814w']['ra'],c['f814w']['dec'],s=1,marker='x',c=c['f814w'][sky],zorder=0)
    plt.scatter(c['f814w']['ra'][zsky8],c['f814w']['dec'][zsky8],s=1,marker='x',c='r')#c['f555w'][sky])
    plt.title('f814w sky = 0, n159e')
    plt.colorbar(label='sky')'''

    if region_shortname == 'n159w':
        c555_orig = SkyCoord(ra=pass_555['ra'] * u.deg, dec=pass_555['dec'] * u.deg)
        c555 = c555_orig.spherical_offsets_by(np.repeat(mean_ra_off,len(c555_orig))*u.arcsec,
                                              np.repeat(mean_dec_off,len(c555_orig))*u.arcsec)
    if region_shortname != 'n159w':
        c555 = SkyCoord(ra=pass_555['ra'] * u.deg, dec=pass_555['dec'] * u.deg)
    c814 = SkyCoord(ra=pass_814['ra'] * u.deg, dec=pass_814['dec'] * u.deg)
    max_sep = 0.2 * u.arcsec
    idx, d2d, d3d = c814.match_to_catalog_sky(c555)
    sep_constraint = d2d < max_sep
    print(np.sum(sep_constraint))
    c814_matches = pass_814[sep_constraint]
    c555_matches = pass_555[idx[sep_constraint]]
    c814_cat = c814_matches[['ra', 'dec', x, y, mag, dmag, snr, shp, rnd, otype, crd,'satur']].to_pandas()
    c814_cat.columns = [cname + '_814' for cname in
                        ['ra', 'dec', 'x', 'y', 'mag', 'dmag', 'snr', 'shp', 'rnd', 'otype', 'crd','satur']]
    c555_cat = c555_matches[['ra', 'dec', x, y, mag, dmag, snr, shp, rnd, otype, crd,'satur']].to_pandas()
    c555_cat.columns = [cname + '_555' for cname in
                        ['ra', 'dec', 'x', 'y', 'mag', 'dmag', 'snr', 'shp', 'rnd', 'otype', 'crd','satur']]
    cmatch_vi = c555_cat.join(c814_cat)
    cmatch_vi['555min814'] = cmatch_vi['mag_555'] - cmatch_vi['mag_814']

    cmatch_vi.to_csv(photdir+region+'new_new_xmatch_0.2sec.555.814_f814ref.csv', index=False)





    xs = cmatch_vi['555min814']
    ys = cmatch_vi['mag_555']


    plt.figure()
    plt.scatter(xs,ys,s=1,alpha=0.9,c=cmatch_vi['satur_814'])
    plt.gca().invert_yaxis()
    plt.xlabel('F555 - F814 [mag]')
    plt.ylabel('F555 [mag]')
    plt.colorbar()


    xbins = np.arange(np.min(xs), np.max(xs), 0.1)
    ybins = np.linspace(np.min(ys), np.max(ys), len(xbins))
    # create fig & plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    hdict = hess_bin(filt1=xs, filt2=ys,
                     xbins=xbins, ybins=ybins,
                     ax=ax, fig=fig, cm='viridis')
    ax.set_xlabel('F555 - F814')
    ax.set_ylabel('F555')
    #plt.savefig('n159s_hess_555_814.png',dpi=300)

    ###########################

    xs = cmatch_vi['555min814']
    ys = cmatch_vi['mag_555']
    '''
    ## for n159s
    d_rcline = (xs-0.3)*(22.7-19) - (ys-19)*(1.75-0.3)
    line_side = d_rcline >= 0
    rc_inds = (ys <= 23) & (ys >= 19) & line_side & (xs <= 2.75) & (xs >= 0.75)
    '''
    d_rcline = (xs-0.9)*(24-20.2) - (ys-20.2)*(2.5-0.9)
    line_side = d_rcline >= 0
    rc_inds = (ys <= 22.5) & (ys >= 19) & line_side & (xs <= 2.75) & (xs >= 1)


    rc_df = cmatch_vi[rc_inds]
    #rc_inds = (cmatch_vi['555min814'] <= 2.25) & (cmatch_vi['555min814'] >= 0.75) & \
    #            (cmatch_vi['mag_555'] <= 20.5) & (cmatch_vi['mag_555'] >= 19)


    plt.figure()
    plt.scatter(rc_df['555min814'], rc_df['mag_555'], s=2, alpha=1,c='r')
    plt.scatter(cmatch_vi['555min814'], cmatch_vi['mag_555'],
                s=2, alpha=0.8,c=line_side,zorder=1)
    plt.gca().invert_yaxis()
    plt.xlabel('F555 - F814 [mag]')
    plt.ylabel('F555 [mag]')
    plt.title('red clump fitting area')
    #plt.plot([0.4, 1.85], [19, 22.7],c='r')
    plt.plot([0.9-1,2.5-1],[18.7-0.5,22.5-0.5],c='r')
    plt.plot([])
    plt.axhline(y=19,c='k')
    plt.axhline(y=22.5,c='k')



    from sklearn import linear_model


    def ransac_rc(X,y):

        # Robustly fit linear model with RANSAC algorithm
        ransac = linear_model.RANSACRegressor(max_trials=500)
        ransac.fit(X, y)
        inlier_mask = ransac.inlier_mask_
        # outlier_mask = np.logical_not(inlier_mask)
        # line_X = np.arange(X.min(), X.max())[:, np.newaxis]
        # line_y_ransac = ransac.predict(line_X)
        #### check fit slope
        # Estimated coefficients
        m, c = float(ransac.estimator_.coef_), float(ransac.estimator_.intercept_)
        #print("\nm={:.3f}, c={:.3f}".format(m, c))

        return [m, c,inlier_mask*1]

    X = rc_df['555min814'].values.reshape(-1,1)
    y = rc_df['mag_555'].values

    nrun = 500
    ran_mc = np.vstack([ransac_rc(X,y) for i in range(nrun)])
    ran_m = ran_mc[:,0]
    ran_c = ran_mc[:,1]
    ran_in = ran_mc[:,2]
    P_in = np.sum(ran_in,axis=0)/nrun

    lw = 2
    line_X = np.arange(X.min(), X.max()-0.5,0.05)#[:, np.newaxis]
    line_Y = np.median(ran_m)*line_X + np.median(ran_c)

    rc_pcheck = P_in >= 0.95

    fig = plt.figure(figsize=(15,7))
    ax = fig.add_subplot(121)
    scats = ax.scatter(X,y,c=P_in,cmap=cmr.ember,s=6,marker='o',label='Red Clump Samples')
    #scats = ax.scatter(X[rc_pcheck],y[rc_pcheck],c='r',s=6,marker='o',label='Red Clump Samples')
    fig.colorbar(scats,ax=ax,label='P(inlier)')
    plt.plot(  line_X, line_Y,   color="cornflowerblue",  linewidth=lw, label="RANSAC regressor")
    plt.scatter(cmatch_vi['555min814'], cmatch_vi['mag_555'], s=1, alpha=1,c='grey',label='Full sample',zorder=0)
    plt.legend(loc="lower left")
    plt.xlabel('F555 - F814 [mag]')
    plt.ylabel('F555 [mag]')
    plt.gca().invert_yaxis()
    plt.title('N159w RANSAC Regression: ')
    #plt.xlim(-0.1,3)
    #plt.ylim(22.5,19)

    ax1 = fig.add_subplot(222)
    ax1.hist(ran_m,bins=30,facecolor='crimson')
    ax1.set_xlabel('Slope')
    ax1.set_title('Slope estimates')
    ax1.axvline(x=np.median(ran_m),c='cornflowerblue',lw=3,\
                label='$R_{555W} =$  %.2f'%np.median(ran_m)+' $\pm$ %.2f'%np.std(ran_m))
    ax1.legend()
    ax2 = fig.add_subplot(224)
    ax2.hist(ran_c,bins=30,facecolor='crimson')
    ax2.axvline(x=np.median(ran_c),c='cornflowerblue',lw=3,
                label='Intercept =  %.2f'%np.median(ran_c)+' $\pm$ %.2f'%np.std(ran_c))
    ax2.legend(loc='upper right')
    ax2.set_xlabel('Intercept')
    ax2.set_title('Intercept estimates')

    fig.tight_layout()

    ##############

    fig = plt.figure(figsize=(9,9))
    ax = fig.add_subplot(111)
    axplt.title('N159S RANSAC for red clump')

    scats = ax.scatter(rc_df['ra_555'],rc_df['dec_555'],c=P_in,\
                       cmap=cmr.ember,s=10,marker='s',label='Red Clump Samples')
    plt.scatter(cmatch_vi['ra_555'],cmatch_vi['dec_555'], s=0.25, alpha=0.25,c='grey',label='Full sample',zorder=0)
    fig.colorbar(scats,ax=ax,label='P(inlier)')
    ax.legend()
    plt.xlabel('RA')
    plt.ylabel('Dec')





######################################################
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