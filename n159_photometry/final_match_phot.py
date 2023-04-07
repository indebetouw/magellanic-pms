from astropy.table import Table
import matplotlib.pyplot as pl
pl.ion()
#pl.clf()
import os, subprocess
import numpy as np
from astropy.io import fits
from astropy.wcs import wcs

region="n159-e"
shortregion = 'n159e'
region_shortname = 'n159e'

#######################################################
#filts=["f555w","f814w"]
filts = ["f125w", "f160w"]
reload=False

mag = 'col16'
dmag= 'col18'
snr = 'col6'
shp = 'col7'
rnd = 'col8'
x   = 'col3'
y   = 'col4'
otype = 'col11'
crd = 'col10'

ref_filt = '_f160ref' # '_f814ref'

#workdir="/Users/remy/cv/magellanic/n159/"

#workdir = '/Users/toneill/N159/photometry/new/n159-e/n159-e-newoptical/'
workdir = '/Users/toneill/N159/photometry/new/'+region+'/'

#workdir = photdir+region+'/'+'n159-e-newoptical/'##.phot0/'

#os.chdir(workdir+region+"/")
os.chdir(workdir)


try:
    if len(c)<=0:
        reload=True
except:
    reload=True

if reload:
    c={}
    for filt in filts:
        # this is of course particular to this project that we did 555,814 with ACS and IR with WFC3
        kind="flc"
        camera="acs"
        catfile = filt+"/"+region+"_"+filt+ref_filt
        shpmax=0.27
        crdmax=1
        
        print(catfile)
        intab=Table.read(catfile,format="ascii")
        # quality cuts from PHAT: this breaks the position match between the pairs of filters 
        good=np.where((intab[snr]>4)*(intab[shp]<shpmax)*(intab[crd]<crdmax))[0]
        #good=np.where((intab[snr]>4))[0]
        c[filt]=intab[good]
        print(len(c[filt]))
        print(c[filt][0:5])

    # get reference file and ra/dec for each filter
    for filt in filts:
        # print('grep img_file '+filt+'/dolparms_'+filt+".multi.txt")
        c[filt+"_ref"] = reffile = filt+"/"+subprocess.getoutput('grep img_file '+filt+'/dolparms_'+filt+".multi.txt").split()[-1]+".fits"
        print(reffile)
        c[filt+"_wcs"] = wcs.WCS(fits.getheader(reffile))
        xpix=c[filt][x]
        ypix=c[filt][y]
        ra,de=c[filt+"_wcs"].wcs_pix2world(xpix,ypix,0)
        c[filt+"_ra"]=ra
        c[filt+"_de"]=de
        print(ra[0:10],de[0:10])

    if region_shortname == 'n159w' and 'f555w' in filts:
        from astropy import units as u
        from astropy.coordinates import SkyCoord

        mean_ra_off = -0.16816
        mean_dec_off = -0.18507

        r555_orig = SkyCoord(ra=c['f555w_ra'] * u.deg, dec=c['f555w_de'] * u.deg)
        r555_fix = r555_orig.spherical_offsets_by(np.repeat(mean_ra_off,len(r555_orig))*u.arcsec,
                                              np.repeat(mean_dec_off,len(r555_orig))*u.arcsec)

        c["f555w_ra"]= r555_fix.ra.value
        c["f555w_de"]= r555_fix.dec.value


    from tqdm import tqdm

    match12=np.zeros(len(c[filts[0]]),dtype=int)-1
    match21=np.zeros(len(c[filts[1]]),dtype=int)-1
    d=0.2 # arcsec

    pbar = tqdm(total=len(c[filts[0]]), unit='star')
    for i in range(len(c[filts[0]])):
        dra=np.absolute(c[filts[0]+"_ra"][i]-c[filts[1]+"_ra"])*np.cos(c[filts[0]+"_de"][i]*np.pi/180)
        dde=np.absolute(c[filts[0]+"_de"][i]-c[filts[1]+"_de"])
        z=np.where( (dra<d)*(dde<d) )[0]
        if len(z)>0:
            d2= dra[z]**2 + dde[z]**2
            zz=z[np.where(d2==d2.min())[0][0]]
            match12[i]=zz
            match21[zz]=i
        pbar.update(1)



#------
pl.figure()
g=pl.gcf()
g.set_size_inches(6,6)
z=np.where(match12>=0)[0]
c1=c[filts[0]]
c2=c[filts[1]]

if filts == ['f555w','f814w']:
    '''z1=z[ np.where((c1[mag][z]<90)*(c2[mag][match12[z]]<90)*
                   (c1[dmag][z]<0.1)*(c2[dmag][match12[z]]<0.1)
                   #(np.absolute(c1[shp][z])<1)*(np.absolute(c2[shp][match12[z]])<1)*
                   #(c1[rnd][z]<3)*(c2[rnd][match12[z]]<3)*
                   #(c1[crd][z]<0.3)*(c2[crd][match12[z]]<0.3)*
                   #(c1[otype][z]<3)*(c2[otype][match12[z]]<3)
                  )[0] ]

    z2=z[ np.where((c1[mag][z]<90)*(c2[mag][match12[z]]<90)*
                   (c1[dmag][z]<0.1)*(c2[dmag][match12[z]]<0.1)*
                   (c1[shp][z]<shpmax)*(c2[shp][match12[z]]<shpmax)*
                   #(c1[rnd][z]<3)*(c2[rnd][match12[z]]<3)*
                   (c1[crd][z]<crdmax)*(c2[crd][match12[z]]<crdmax)
                   #(c1[otype][z]<3)*(c2[otype][match12[z]]<3)
                  )[0] ]'''

    z3=z[ np.where((c1[mag][z]<90)*(c2[mag][match12[z]]<90)*
                   (c1[dmag][z]<0.1)*(c2[dmag][match12[z]]<0.1)*
                   (np.absolute(c1[shp][z])<1)*(np.absolute(c2[shp][match12[z]])<1)*
                   (c1[rnd][z]<3)*(c2[rnd][match12[z]]<3)*
                   (c1[crd][z]<0.3)*(c2[crd][match12[z]]<0.3)*
                   (c1[otype][z]<3)*(c2[otype][match12[z]]<3)
                  )[0] ]

if filts == ['f125w','f160w']:

    z3 = z[np.where( (c1[mag][z] < 90) * (c2[mag][match12[z]] < 90) *
                     (c1[snr][z] >= 10) * (c2[snr][match12[z]] >= 10) *
           (c1[crd][z] <= 0.48) * (c2[crd][match12[z]] <= 0.48) *
           (c1[shp][z] > -0.6) * (c2[shp][match12[z]] > -0.6))[0] ]


#pl.plot(c1[mag][z1]-c2[mag][match12[z1]],c1[mag][z1],'.',label='all',markersize=0.5)

#pl.plot(c1[mag][z2]-c2[mag][match12[z2]],c1[mag][z2],'.',label='PHAT',markersize=0.5)

pl.plot(c1[mag][z3]-c2[mag][match12[z3]],c1[mag][z3],'.',label='RI',markersize=2)#,c='g',alpha=0.8)

pl.xlabel(filts[0] + "-" + filts[1])
pl.ylabel(filts[0])
if camera == "acs":
    pl.xlim(-0.5, 3.5)
    pl.ylim(29, 13)
else:
    pl.xlim(-0.5, 1.5)
    pl.ylim(26, 13)

pl.legend(loc="best", prop={"size": 10})
pl.title(f'{region}')
pl.tight_layout()

#################################################################
c555_matches = c1[z3].to_pandas()
c555_match_ra = c[filts[0]+'_ra'][z3]
c555_match_dec = c[filts[0]+'_de'][z3]

c814_matches = c2[match12[z3]].to_pandas()
c814_match_ra = c[filts[1]+'_ra'][match12[z3]]
c814_match_dec = c[filts[1]+'_de'][match12[z3]]

shortfilts = [filts[0][1:4],filts[1][1:4]]

c814_cat = c814_matches[[x, y, mag, dmag, snr, shp, rnd, otype, crd]]
c814_cat.columns = [cname +'_'+shortfilts[1] for cname in
                    [ 'x', 'y', 'mag', 'dmag', 'snr', 'shp', 'rnd', 'otype', 'crd']]
c555_cat = c555_matches[[x, y, mag, dmag, snr, shp, rnd, otype, crd]]
c555_cat.columns = [cname + '_'+shortfilts[0] for cname in
                    ['x', 'y', 'mag', 'dmag', 'snr', 'shp', 'rnd', 'otype', 'crd']]

cmatch_vi = c555_cat.join(c814_cat)
cmatch_vi['ra_'+filts[0]] =c555_match_ra
cmatch_vi['dec_'+filts[0]] =c555_match_dec
cmatch_vi['ra_'+filts[1]] =c814_match_ra
cmatch_vi['dec_'+filts[1]] =c814_match_dec

cmatch_vi[shortfilts[0]+'min'+shortfilts[1]] = cmatch_vi['mag_'+shortfilts[0]] - cmatch_vi['mag_'+shortfilts[1]]

savephotdir = '/Users/toneill/N159/photometry/reduced/'

cmatch_vi.to_csv(savephotdir+region+'_reduce_'+filts[0]+'_'+filts[1]+'.phot.csv',index=False)

pl.figure()
pl.scatter(cmatch_vi[shortfilts[0]+'min'+shortfilts[1]],cmatch_vi['mag_'+shortfilts[0]],s=0.5,c='r')
pl.gca().invert_yaxis()


#################################################################

# match ir & visible
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np

region = 'n159-w'
savephotdir = '/Users/toneill/N159/photometry/reduced/'
finphotdir = '/Users/toneill/N159/photometry/FINAL_PHOT/'+region+'/'

vi_df = pd.read_csv(savephotdir+region+'_reduce.phot.csv')
ir_df = pd.read_csv(savephotdir+region+'_reduce_f125w_f160w.phot.csv')
## rename ir position names
ir_df = ir_df.rename(columns={'ra_f125w':'ra_125','dec_f125w':'dec_125',
                              'ra_f160w':'ra_160','dec_f160w':'dec_160'})

c814 = SkyCoord(ra=vi_df['ra_814'] * u.deg, dec=vi_df['dec_814'] * u.deg)
c160 = SkyCoord(ra=ir_df['ra_160'] * u.deg, dec=ir_df['dec_160'] * u.deg)

max_sep = 0.5* u.arcsec
idx, d2d, d3d = c814.match_to_catalog_sky(c160)
sep_constraint = d2d < max_sep
print(np.sum(sep_constraint))
c814_matches = vi_df[sep_constraint].reset_index()
c160_matches = ir_df.iloc[idx[sep_constraint]].reset_index()
cmatch_irvi = c160_matches.join(c814_matches,lsuffix='_ir',rsuffix='_vis')

cmatch_irvi.to_csv(finphotdir+region+'_phot_vis.ir.csv',index=False)
vi_df.to_csv(finphotdir+region+'_phot_vis.csv',index=False)
ir_df.to_csv(finphotdir+region+'_phot_ir.csv',index=False)



###################################################

region = 'n159-all'
savephotdir2 = '/Users/toneill/N159/photometry/FINAL_PHOT/'
finphotdir = '/Users/toneill/N159/photometry/FINAL_PHOT/'+region+'/'

fuse = 'ir'

#for fuse in ['vis','ir','vis.ir']:
edf = pd.read_csv(savephotdir2+'n159-e/'+f'n159-e_phot_{fuse}.csv')
wdf = pd.read_csv(savephotdir2+'n159-w/'+f'n159-w_phot_{fuse}.csv')
sdf = pd.read_csv(savephotdir2+'n159-s/'+f'n159-s_phot_{fuse}.csv')
alldf =pd.read_csv(finphotdir+region+f'_phot_{fuse}.csv')

print(f'\n {fuse}')
print(f'E: {len(edf)}')
print(f'W: {len(wdf)}')
print(f'S: {len(sdf)}')
print(f'All: {len(alldf)}')



if 'in_n159e' in wdf.columns:
    wdf = wdf.drop(columns='in_n159e')

cfilts = {'vis':'814','ir':'160','vis.ir':'814'}
cfilt = cfilts[fuse]

ce = SkyCoord(ra=edf[f'ra_{cfilt}'] * u.deg, dec=edf[f'dec_{cfilt}'] * u.deg)
cw = SkyCoord(ra=wdf[f'ra_{cfilt}'] * u.deg, dec=wdf[f'dec_{cfilt}'] * u.deg)

max_sep = 0.2* u.arcsec
idx, d2d, d3d = cw.match_to_catalog_sky(ce)
sep_constraint = d2d < max_sep
print(np.sum(sep_constraint))
cw_matches = wdf[sep_constraint].reset_index()
ce_matches = edf.iloc[idx[sep_constraint]].reset_index()
cmatch_ew = cw_matches.join(ce_matches,lsuffix='_w',rsuffix='_e')

cw_noe = wdf[~sep_constraint].reset_index()
cw_noe['region'] = np.repeat('n159w',len(cw_noe))
edf['region'] = np.repeat('n159e',len(edf))
sdf['region'] = np.repeat('n159s',len(sdf))

comb_ew = pd.concat([edf,cw_noe],ignore_index=True)
comb_ew = comb_ew.drop(columns='index')

comb_ews = pd.concat([comb_ew,sdf],ignore_index=True)
comb_ews.to_csv(finphotdir+region+f'_phot_{fuse}.csv',index=False)

wdf['in_n159e'] = sep_constraint
wdf.to_csv(savephotdir2+'n159-w/'+f'n159-w_phot_{fuse}.csv',index=False)

##############################################






