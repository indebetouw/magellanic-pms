import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from astropy.wcs import wcs
from load_data import load_phot

fuse = 'vis.ir'
all_vi = load_phot(region='n159-all',fuse=fuse)


wdir = '/Users/toneill/N159/gaia/'


nn_ids = np.loadtxt(wdir+'J_A+A_669_A91/class.dat')


from astroquery.vizier import Vizier
from astropy.table import Table
from astropy.coordinates import SkyCoord
from astropy import units as u


viz = Vizier(columns=['**', '_RAJ2000', '_DEJ2000'])
viz.ROW_LIMIT = 20e6

table_id = 'J/A+A/669/A91/'
tcat = viz.get_catalogs(table_id)[0]
print(tcat.colnames)

lmc_names = tcat['GaiaDR3'].value


gaia_call = Table.read(wdir+'gaia.fit')
call_names = gaia_call['Source'].value

from tqdm import tqdm

match_idx = np.full(len(call_names),np.nan)

pbar = tqdm(total=len(call_names), unit='star')
for i in range(len(call_names)):
    if call_names[i] in lmc_names:
        #print('yay')
        match_idx[i], = np.where(lmc_names == call_names[i])
    pbar.update(1)

match_df = gaia_call[~np.isnan(match_idx)]
match_df['pLMC'] = tcat['P'].value[np.array(match_idx[~np.isnan(match_idx)],dtype='int')]
Table.to_pandas(match_df).to_csv(wdir+'match_df.csv',index=False)

lmc_df = match_df[match_df['pLMC'] >= 0.5]
Table.to_pandas(lmc_df).to_csv(wdir+'match_lmc_df.csv',index=False)


plt.figure(figsize=(8,8))
plt.scatter(lmc_df['RAJ2000'],lmc_df['DEJ2000'],c='r')#,c=lmc_df['pLMC']cmap='RdYlBu',s=1,vmin=0,vmax=1)
plt.scatter(all_vi['ra_555'],all_vi['dec_555'],c='royalblue',zorder=0)
plt.gca().invert_xaxis()


lmc_sc = SkyCoord(lmc_df['RAJ2000'],lmc_df['DEJ2000'],frame='fk5')
phot_sc = SkyCoord(all_vi['ra_555']*u.deg,all_vi['dec_555']*u.deg,frame='fk5')

max_sep = 0.5* u.arcsec
idx, d2d, d3d = phot_sc.match_to_catalog_sky(lmc_sc)
sep_constraint = d2d < max_sep
print(np.sum(sep_constraint))
c814_matches = all_vi[sep_constraint].reset_index()
c160_matches = Table.to_pandas(lmc_df[idx[sep_constraint]])#.reset_index()
cmatch_irvi = c160_matches.join(c814_matches,lsuffix='_gaia',rsuffix='_hst')

cmatch_irvi.to_csv(wdir+'gaia_xmatch_phot_vis.csv',index=False)


plt.figure(figsize=(8,8))
plt.scatter(cmatch_irvi['RAJ2000'],cmatch_irvi['DEJ2000'],c='r')#,c=lmc_df['pLMC']cmap='RdYlBu',s=1,vmin=0,vmax=1)
plt.scatter(all_vi['ra_555'],all_vi['dec_555'],c='royalblue',zorder=0)
plt.gca().invert_xaxis()
