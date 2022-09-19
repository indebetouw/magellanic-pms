
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import pandas as pd
import vaex as vx
import os
from astropy.table import Table
from astropy.wcs import wcs
import subprocess

photdir = '/Users/toneill/N159/photometry/'

f814_cols = pd.read_csv(photdir+'n159s.phot0/f814w/n159s_f814w_f814ref.columns',sep='\\',header=None)[0]

region = "n159s"
filts = ["f555w", "f814w"]
filts = ["f125w", "f160w"]

workdir = photdir+'n159s.phot0/'


#os.chdir(workdir + region + "/")

c = {}
for filt in filts:
    if filt == "f125w" or filt == "f160w":
        kind = "flt"
        camera = "wfc3"
        catfile = filt + "/" + region + "_" + filt + "_f160ref"
    else:
        # this is of course particular to this project that we did UVO with ACS and IR with WFC3
        kind = "flc"
        camera = "acs"
        catfile = filt + "/" + region + "_" + filt + "_f814ref"

    print(catfile)
    c[filt] = Table.read(workdir+catfile, format="ascii")
    print(len(c[filt]))
    print(c[filt][0:5])

# column definitions
mag = 'col16'
dmag= 'col18'
snr = 'col6'
shp = 'col7'
rnd = 'col8'
x   = 'col3'
y   = 'col4'
otype = 'col11'
crd = 'col10'

ref_head = fits.open(photdir+'/n159s.phot0/f814w/n159s_f814w_f814ref.1.res.fits')[0].header
ref_wcs = wcs.WCS(ref_head)

'''
RA_TARG =   8.501666666667E+01 / right ascension of the target (deg) (J2000)    
DEC_TARG=  -6.984888888889E+01 / declination of the target (deg) (J2000)       

WCSNAMEO= 'OPUS    '           / Coordinate system title                        
WCSAXESO=                    2 / Number of coordinate axes               
CRPIX1O =               2100.0 / Pixel coordinate of reference point            
CRPIX2O =               1024.0 / Pixel coordinate of reference point            
CUNIT1O = 'deg'                / Units of coordinate increment and value        
CUNIT2O = 'deg'                / Units of coordinate increment and value        
CTYPE1O = 'RA---TAN'           / Right ascension, gnomonic projection           
CTYPE2O = 'DEC--TAN'           / Declination, gnomonic projection               
CRVAL1O =       85.05637933215 / [deg] Coordinate value at reference point      
CRVAL2O =      -69.85704854218 / [deg] Coordinate value at reference point      

WCSNAMEA= 'IDC_0461802dj-HSC30' / Coordinate system title                       
WCSAXESA=                    2 / Number of coordinate axes                      
CRPIX1A =               2048.0 / Pixel coordinate of reference point            
CRPIX2A =               1024.0 / Pixel coordinate of reference point            
CUNIT1A = 'deg'                / Units of coordinate increment and value        
CUNIT2A = 'deg'                / Units of coordinate increment and value        
CTYPE1A = 'RA---TAN-SIP'       / TAN (gnomonic) projection + SIP distortions    
CTYPE2A = 'DEC--TAN-SIP'       / TAN (gnomonic) projection + SIP distortions    
CRVAL1A =      85.057237135068 / [deg] Coordinate value at reference point      
CRVAL2A =     -69.856374724365 / [deg] Coordinate value at reference point      
CRDER1A =       4.372601452571 / [deg] Random error in coordinate               
CRDER2A =      4.5928129828215 / [deg] Random error in coordinate               
RADESYSA= 'ICRS'               / Equatorial coordinate system              

WCSNAME = 'Gaia'               / Coordinate system title                        
WCSAXES =                    2 / Number of coordinate axes                      
CRPIX1  =               2048.0 / Pixel coordinate of reference point            
CRPIX2  =               1024.0 / Pixel coordinate of reference point            
CUNIT1  = 'deg'                / Units of coordinate increment and value        
CUNIT2  = 'deg'                / Units of coordinate increment and value        
CTYPE1  = 'RA---TAN-SIP'       / TAN (gnomonic) projection + SIP distortions    
CTYPE2  = 'DEC--TAN-SIP'       / TAN (gnomonic) projection + SIP distortions    
CRVAL1  =      85.057402667938 / [deg] Coordinate value at reference point      
CRVAL2  =     -69.856377840029 / [deg] Coordinate value at reference point         
'''

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





