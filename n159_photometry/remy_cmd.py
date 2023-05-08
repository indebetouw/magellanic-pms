from astropy.table import Table
import matplotlib.pyplot as pl
pl.ion()
#pl.clf()
import os, subprocess
import numpy as np
from astropy.io import fits
from astropy.wcs import wcs

let = 'w'
region="n159-"+let
shortregion = 'n159'+let
region_shortname = 'n159'+let


region = 'off-point'
shortregion = 'off'
region_shortname ='off'


#######################################################
filts=["f555w","f814w"]
#filts = ["f125w", "f160w"]
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

if filts ==  ["f125w", "f160w"]:
    ref_filt = '_f160ref'
if filts == ["f555w","f814w"]:
        ref_filt = '_f814ref'

workdir = '/Users/toneill/N159/photometry/new/'+region+'/'
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

        if filts == ["f555w","f814w"]:
            shpmax=0.27
            crdmax=1
        if filts == ["f125w", "f160w"]:
            shpmax = 0.34
            crdmax = 1
        
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

    '''cat1 = SkyCoord(ra=c[filts[0]+"_ra"]*u.deg,dec=c[filts[0]+"_de"]*u.deg)
    cat2 = SkyCoord(ra=c[filts[1]+"_ra"]*u.deg,dec=c[filts[1]+"_de"]*u.deg)
    match , d2d, d3d = cat1.match_to_catalog_sky(cat2)
    sep_crit = d2d < 0.2*u.arcsec
    np.sum(sep_crit)
    off_to = cat1.spherical_offsets_to(cat2[match])
    plt.figure(figsize=(6,6))
    plt.scatter(off_to[0],off_to[1],s=0.1,alpha=0.1)
    plt.axvline(x=0,c='k')
    plt.axhline(y=0,c='k')'''

#------
z=np.where(match12>=0)[0]
c1=c[filts[0]]
c2=c[filts[1]]
if filts == ['f555w','f814w']:
    '''z3 = z[np.where((c1[mag][z] < 90) * (c2[mag][match12[z]] < 90) *
                (c1[snr][z] >= 10) * (c2[snr][match12[z]] >= 10) *
                (np.absolute(c1[shp][z]) < 1) * (np.absolute(c2[shp][match12[z]]) < 1) *
                (c1[rnd][z] < 3) * (c2[rnd][match12[z]] < 3) *
                (c1[crd][z] < 0.3) * (c2[crd][match12[z]] < 0.3) *
                (c1[otype][z] < 3) * (c2[otype][match12[z]] < 3))[0]]  # '''

    magmax1 = 28
    magmax2 = 28
    colormin = -1
    colormax = 5
    snrmin = 10
    otypemax = 3
    crdmax = 0.3
    z3 = z[np.where((c1[mag][z] < magmax1) * (c2[mag][match12[z]] < magmax2) *
                   (c1[mag][z]-c2[mag][match12[z]] > colormin)*
                    (c1[mag][z] - c2[mag][match12[z]] < colormax)*
                (c1[snr][z] >= snrmin) * (c2[snr][match12[z]] >= snrmin) *
                    (c1[shp][z]**2 < 0.075) * (c2[shp][match12[z]]**2 < 0.075)*
                    (c1[rnd][z] < 3) * (c2[rnd][match12[z]] < 3) *
                (c1[crd][z] < crdmax) * (c2[crd][match12[z]] < crdmax) *
                (c1[otype][z] < otypemax) * (c2[otype][match12[z]] < otypemax))[0]]
    print(f'z3: {len(z3)}')

if filts == ['f125w','f160w']:
    snrmin = 10
    crdmax = 0.3#48
    otypemax = 3
    colormin = -1
    colormax = 5
    magmax1 = 26
    magmax2 = 26
    z3 = z[np.where( (c1[mag][z] < magmax1) * (c2[mag][match12[z]] < magmax2) *
                     (c1[mag][z] - c2[mag][match12[z]] > colormin) *
                     (c1[mag][z] - c2[mag][match12[z]] < colormax) *
                     (c1[snr][z] >= snrmin) * (c2[snr][match12[z]] >= snrmin) *
           (c1[crd][z] < crdmax) * (c2[crd][match12[z]] < crdmax) *
           (c1[shp][z]**2 < 0.12) * (c2[shp][match12[z]]**2 < 0.12)*
                     (c1[otype][z]<otypemax)*(c2[otype][match12[z]]<otypemax)
                 * (c1[rnd][z] < 3) * (c2[rnd][match12[z]] < 3)
                     )[0] ]
    print(f'z3: {len(z3)}')

pl.figure(figsize=(6,6))
pl.scatter(c1[mag][z3]-c2[mag][match12[z3]],c1[mag][z3],label='RI',s=1,alpha=0.5)
pl.xlabel(filts[0] + "-" + filts[1])
pl.ylabel(filts[0])
pl.gca().invert_yaxis()
pl.title(f'{region}')
pl.tight_layout()
pl.xlim(-1,5)
pl.ylim(29,12)


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





#################################################################

# match ir & visible
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt
import numpy as np

region = 'off-point'
savephotdir = '/Users/toneill/N159/photometry/reduced/'
finphotdir = '/Users/toneill/N159/photometry/FINAL_PHOT/'+region+'/'

vi_df = pd.read_csv(savephotdir + region + '_reduce_f555w_f814w.phot.csv')
vi_df = vi_df.rename(columns={'ra_f555w':'ra_555','dec_f555w':'dec_555',
                              'ra_f814w':'ra_814','dec_f814w':'dec_814'})
#else:
#    vi_df = pd.read_csv(savephotdir+region+'_reduce.phot.csv')
ir_df = pd.read_csv(savephotdir+region+'_reduce_f125w_f160w.phot.csv')
## rename ir position names
ir_df = ir_df.rename(columns={'ra_f125w':'ra_125','dec_f125w':'dec_125',
                              'ra_f160w':'ra_160','dec_f160w':'dec_160'})
#ir_df = ir_df[(ir_df['crd_160']<0.3)&(ir_df['crd_125']<0.3)]
print(f'vi: {len(vi_df)}')
print(f'ir: {len(ir_df)}')

print(len(ir_df)/len(vi_df))


c814 = SkyCoord(ra=vi_df['ra_814'] * u.deg, dec=vi_df['dec_814'] * u.deg)
c160 = SkyCoord(ra=ir_df['ra_160'] * u.deg, dec=ir_df['dec_160'] * u.deg)

max_sep = 0.2* u.arcsec

'''
idx, d2d, d3d = c814.match_to_catalog_sky(c160)
sep_constraint = d2d < max_sep
print(f'match: {np.sum(sep_constraint)}')
c814_matches = vi_df[sep_constraint].reset_index()
c160_matches = ir_df.iloc[idx[sep_constraint]].reset_index()
cmatch_irvi = c160_matches.join(c814_matches,lsuffix='_ir',rsuffix='_vis')
'''

idx, d2d, d3d = c160.match_to_catalog_sky(c814)
sep_constraint = d2d < max_sep
print(f'match: {np.sum(sep_constraint)}')

c814_matches = vi_df.iloc[idx[sep_constraint]].reset_index()
c160_matches = ir_df[sep_constraint].reset_index()
cmatch_irvi = c160_matches.join(c814_matches,lsuffix='_ir',rsuffix='_vis')

print(len(cmatch_irvi))
#print(len(cmatch_irvi[cmatch_irvi['rnd_125']<3])/len(c160[ir_df['rnd_125']<3]))
print(len(np.unique(idx[sep_constraint]))/np.sum(sep_constraint))
print(len(cmatch_irvi)/len(c160))



len(np.unique(idx[sep_constraint]))

cmatch_irvi.to_csv(finphotdir+region+'_phot_vis.ir.csv',index=False)
vi_df.to_csv(finphotdir+region+'_phot_vis.csv',index=False)
ir_df.to_csv(finphotdir+region+'_phot_ir.csv',index=False)


#############
off_to = c160.spherical_offsets_to(c814[idx])

plt.figure(figsize=(6,6))
plt.scatter(off_to[0],off_to[1],c='royalblue',s=0.2,zorder=0)
plt.scatter(off_to[0][sep_constraint],off_to[1][sep_constraint],c='r',s=0.2,zorder=2)
plt.title(f'{region}')
plt.axvline(x=0,c='k')
plt.axhline(y=0,c='k')

plt.figure()
plt.scatter(cmatch_irvi['555min814'],cmatch_irvi['mag_555'],s=0.5)
plt.gca().invert_yaxis()

plt.figure()
plt.scatter(cmatch_irvi['125min160'],cmatch_irvi['mag_160'],s=0.5)
plt.gca().invert_yaxis()

#######

old_vdf= pd.read_csv(finphotdir+region+'_phot_vis.csv')
old_irdf= pd.read_csv(finphotdir+region+'_phot_ir.csv')
old_wdf = pd.read_csv(finphotdir+region+'_phot_vis.ir.csv')
plt.figure()
plt.scatter(old_wdf['555min814'],old_wdf['mag_555'],s=0.5)
plt.gca().invert_yaxis()


###################################################
############## MATCH N159E AND W


region = 'n159-all'
savephotdir2 = '/Users/toneill/N159/photometry/FINAL_PHOT/'
finphotdir = '/Users/toneill/N159/photometry/FINAL_PHOT/'+region+'/'


fuse = 'vis.ir'

#for fuse in ['vis','ir','vis.ir']:
edf = pd.read_csv(savephotdir2+'n159-e/'+f'n159-e_phot_{fuse}.csv')
wdf = pd.read_csv(savephotdir2+'n159-w/'+f'n159-w_phot_{fuse}.csv')
sdf = pd.read_csv(savephotdir2+'n159-s/'+f'n159-s_phot_{fuse}.csv')
alldf =pd.read_csv(finphotdir+region+f'_phot_{fuse}.csv')

print(f'\n {fuse}')
print(f'E: {len(edf)}')
print(f'W: {len(wdf)}')
print(f'S: {len(sdf)}')
print(f'All, repeats: {len(edf)+len(wdf)+len(sdf)}')
#print(f'All: {len(alldf)}')



if 'in_n159e' in wdf.columns:
    wdf = wdf.drop(columns='in_n159e')

cfilts = {'vis':'814','ir':'160','vis.ir':'814'}
cfilt = cfilts[fuse]

ce = SkyCoord(ra=edf[f'ra_{cfilt}'] * u.deg, dec=edf[f'dec_{cfilt}'] * u.deg)
cw = SkyCoord(ra=wdf[f'ra_{cfilt}'] * u.deg, dec=wdf[f'dec_{cfilt}'] * u.deg)

max_sep =0.2* u.arcsec
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



##############################
##############################################
##############################################

region = 'n159-all'
finphotdir = '/Users/toneill/N159/photometry/FINAL_PHOT/'+region+'/'

all_vis=pd.read_csv(finphotdir+region+f'_phot_vis.csv')
all_ir=pd.read_csv(finphotdir+region+f'_phot_ir.csv')

cfilt1 = '814'
cfilt2 = '160'
ce = SkyCoord(ra=all_vis[f'ra_{cfilt1}'] * u.deg, dec=all_vis[f'dec_{cfilt1}'] * u.deg)
cw = SkyCoord(ra=all_ir[f'ra_{cfilt2}'] * u.deg, dec=all_ir[f'dec_{cfilt2}'] * u.deg)

max_sep =0.5* u.arcsec
idx, d2d, d3d = cw.match_to_catalog_sky(ce)
sep_constraint = d2d < max_sep
print(np.sum(sep_constraint))

off_to = cw.spherical_offsets_to(ce[idx])

plt.figure(figsize=(6,6))
plt.scatter(off_to[0],off_to[1],c='royalblue',s=0.2,zorder=0)
plt.scatter(off_to[0][sep_constraint],off_to[1][sep_constraint],c='orange',s=0.2,zorder=1)
plt.title(f'{region}')
plt.axvline(x=0,c='k')
plt.axhline(y=0,c='k')


#all_vi = load_phot(region='n159-all',fuse='vis.ir')





##############################################
##############################################
##############################################
from sklearn.cluster import DBSCAN

# set up plotting colors for dbscan
colors = ['royalblue', 'tab:orange', 'orangered', 'tab:red', 'm', \
          'hotpink', 'tab:cyan', 'tab:purple', 'gold',
          'limegreen', 'deeppink', 'slateblue','maroon','cornflowerblue',
          'firebrick','navy',  'r', 'darkgrey']
vectorizer = np.vectorize(lambda x: colors[x % len(colors)])

min_samples = 1000
eps= 9e-6

from sklearn.neighbors import NearestNeighbors

x = off_to[0].value
y = off_to[1].value

neight = NearestNeighbors(n_neighbors=min_samples)
nbrs = neight.fit(pd.DataFrame([x,y]).T)
distances,inds = nbrs.kneighbors(pd.DataFrame([x,y]).T)
distances = np.sort(distances,axis=0)
#distances = distances[:,1]

# Choose approximate point where curve becomes smooth - currently
# by eye, could be improved
plt.figure()
plt.plot(distances[:,1])#,marker='v',markersize=3)
plt.xlabel('Source Number')
plt.ylabel('Nearest Neighbor Distance')
plt.title('Nearest Neighbor Distances of Sources')


def run_dbscan(x,y,min_samples=100,eps=0.005):
    m = DBSCAN(eps=eps,min_samples=min_samples)
    m.fit(pd.DataFrame([x,y]).T)
    clusters = m.labels_
    #print(np.unique(clusters))
    print(len(np.unique(clusters)))

    fig = plt.figure(figsize=(6,6))#figsize=(13.5, 6.5))
    ax = fig.add_subplot(111)
    s = ax.scatter(x,y,   c = vectorizer(clusters), s = 0.5)
    ax.axvline(x=0, c='k')
    ax.axhline(y=0, c='k')

    return clusters


clus = run_dbscan(off_to[0].value,off_to[1].value,min_samples=5000,eps=4e-6)
print(np.sum(clus!=-1))

match_clus =np.where( clus==0)[0]
np.mean([off_to[0].value[match_clus],off_to[1].value[match_clus]],axis=1)






##############################################
##############################################
##############################################


plt.figure(figsize=(6,6))
plt.scatter(cmatch_ew['ra_814_w']-cmatch_ew['ra_814_e'],
            cmatch_ew['dec_814_w']-cmatch_ew['dec_814_e'],s=3)
plt.axvline(x=0,c='grey')
plt.axhline(y=0,c='grey')


plt.figure(figsize=(6,6))
plt.scatter(cmatch_ew['mag_555_w']/cmatch_ew['mag_555_e'],
            cmatch_ew['mag_814_w']/cmatch_ew['mag_814_e'],s=1,alpha=0.5)
plt.axvline(x=1,c='grey')
plt.axhline(y=1,c='grey')



plt.figure()
useinds = all_vi['region'] == 'n159w'
plt.scatter(all_vi['ra_814'][useinds],all_vi['dec_814'][useinds],alpha=1,s=1,c='b')
useinds = all_vi['region'] == 'n159e'
plt.scatter(all_vi['ra_814'][useinds],all_vi['dec_814'][useinds],alpha=0.3,c='orange',s=1)


###

#cmatch_irvi = pd.read_csv(savephotdir+region+'_reduce_visir.phot.csv')

plt.figure(figsize=(6,6))
plt.scatter(cmatch_irvi['ra_160']-cmatch_irvi['ra_814'],
            cmatch_irvi['dec_160']-cmatch_irvi['dec_814'],s=3)
plt.axvline(x=0,c='grey')
plt.axhline(y=0,c='grey')




plt.figure()
#plt.scatter(vi_df['555min814'],vi_df['mag_555'],s=0.4,label='All visible')
plt.scatter(cmatch_irvi['555min814'],cmatch_irvi['mag_555'],s=0.4,c='r',label='Match in IR')
plt.xlabel('555 - 814')
plt.ylabel('555')
plt.legend()
plt.gca().invert_yaxis()
plt.title(region)

plt.figure()
#plt.scatter(ir_df['125min160'],ir_df['mag_125'],s=0.4,label='All IR')
plt.scatter(cmatch_irvi['125min160'],cmatch_irvi['mag_125'],s=0.4,c='r',label='Match in visible')
plt.xlabel('125 - 160')
plt.ylabel('125')
plt.legend(loc='upper right')
plt.gca().invert_yaxis()
plt.title(region)
#plt.xlim(-2,3)
#plt.ylim(26,16)

#################























#filts=["f555w","f814w"]
filts = ["f814w", "f160w"]
shortfilts = [filts[0][1:4],filts[1][1:4]]


match12 = np.zeros(len(vi_df), dtype=int) - 1
match21 = np.zeros(len(ir_df), dtype=int) - 1
d = 0.2  # arcsec

for i in range(len(vi_df)):
    dra = np.absolute(vi_df['ra_'+shortfilts[0]][i] -ir_df['ra_'+filts[1]]) * np.cos(vi_df['dec_'+shortfilts[0]][i] * np.pi / 180)
    dde = np.absolute(vi_df['dec_'+shortfilts[0]][i] - ir_df['dec_'+filts[1]])
    z = np.where((dra < d) * (dde < d))[0]
    if len(z) > 0:
        d2 = dra[z] ** 2 + dde[z] ** 2
        zz = z[np.where(d2 == d2.min())[0][0]]
        match12[i] = zz
        match21[zz] = i

z=np.where(match12>=0)[0]
#c1=c[filts[0]]
#c2=c[filts[1]]

vi_imatch = vi_df[match12>=0]
ir_vmatch = ir_df[match21>=0]



pl.plot(vi_df['555min814'].values[z],vi_df['mag_555'].values[match12[z]],'.',label='all',markersize=0.5)
pl.gca().invert_yaxis()

