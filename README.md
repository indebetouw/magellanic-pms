## magellanic-pms
PMS selection and analysis, generally using HST data

### data too large for github:
- ALMA images: https://virginia.app.box.com/folder/137686160073
- photometry catalogs: https://virginia.app.box.com/folder/137685034565
  - HTTP.2015_10_20.1.astro full HTTP catalog from https://archive.stsci.edu/prepds/30dor/Preview/observations.html
  - Ksoll selections from https://ui.adsabs.harvard.edu/abs/2018MNRAS.479.2389K/abstract
- TADA files https://www.dropbox.com/home/TadaFiles

### in this repo:
- code/kNN_extinction_map.py: take list of positions and AVs e.g. from UMS, and make nearest neighbor map
- code/make_Hess.py: take mags in two filters, e.g. V and I, and make a Hess diagram
- code/make_2dkde.py: take any two spatial coordinates, e.g. RA and Dec of PMS stars, and make 2d kernel density estimation
