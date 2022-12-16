









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

hst_phot = pd.read_csv(photdir+'HSC-9_19_2022.csv')

h_555 = hst_phot['A_F555W'].values
h_814 = hst_phot['A_F814W'].values

plt.figure()
plt.gca().invert_yaxis()
plt.scatter(h_555-h_814,h_814,s=0.25,alpha=0.5,c='royalblue')
plt.xlabel('V - I')
plt.ylabel('I')
plt.show()


plt.figure()
plt.scatter(hst_phot['MatchRA'],hst_phot['MatchDec'],s=0.25,alpha=0.5,c='royalblue')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.show()



