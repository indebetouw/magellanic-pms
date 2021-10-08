# create extinction map from k nearest neighbor e.g. UMS star list
# 20210518 added single point function to query

'''
desired call:
    av0   = np.array([kNN_extinction(URA,UDec,UAV,eps,nnear,ri,di) \
                  for ri,di in zip(RA[z0],Dec[z0])])   

'''

def weight_NN(dists,eps=1):
    
    # following equations 1 and 2 in Ksoll2018
    import numpy as np
    wt0 = 1/(dists**2 + eps**2)
    wt = wt0 / np.sum(wt0)
    
    return wt

def kNN_regr_Av(UCoords,tCoords,UAV,eps,nnear,ncore=7):
    
    # perform KNN regression to predict Av
    # more efficient than brute force search & parallel proc if want
    
    # using KD tree because more efficient than ball tree in low-dimensions
    # (thru testing, here ball tree takes ~2x as long)
    from sklearn.neighbors import KNeighborsRegressor
     
    nreg = KNeighborsRegressor(n_neighbors=nnear,weights=weight_NN,
                               n_jobs=ncore,algorithm='kd_tree').fit(UCoords,UAV)
    predAv = nreg.predict(tCoords)
    #nnD, nnIDX = nreg.kneighbors(testCoords)

    return predAv

def kNN_extinction(rastar,destar,avstar,eps,nnear,r,d):
     import numpy as np
     
     cosdec=np.cos(d*np.pi/180)
     dists=np.linalg.norm([(rastar-r)*cosdec,destar-d],axis=0)
     nearest=np.argsort(dists)[0:nnear]
     di = dists[nearest]
     wt0 = 1/(di**2+eps**2) # non-normalized weights
     wt = wt0 / np.sum(wt0)

     if min(di)<2*eps:
          return(np.sum(wt*avstar[nearest]))
     else:
          return(0)
     

def kNN_extinction_map(rastar,destar,avstar     # training set
                       ,ra_range, de_range # map limits, degrees
                       ,outroot="avmap" # will make outroot.fits and outroot.png
                       ,cell=2.     # map cell size in arcsec
                       ,epsilon=5.  # map smoothing param in arcsec
                       ,nnear=20    # number of nearest neighbors
                       ):
     """
     calculate av map from nnear nearest neighbors, 
     over an extent ra_range,de_range, with cell size cell,
     weighted by 1/(distance**2+epsilon**2)
     write outroot.fits with WCS
     write outroot.png 
     return (avmap,wcs)
     """

     import numpy as np
     from astropy import wcs
     from astropy.io import fits
     import matplotlib.pyplot as pl
     
     # cell size for map, deg
     dx = cell/3600
     
     # smoothing parameter, deg
     eps = epsilon/3600
     
     # map range:
     xra=ra_range
     yra=de_range
     
     #========================================================================
     
     cosdec=np.cos(np.mean(yra)*np.pi/180)
     
     # map dimensions:
     nx=int(np.ceil((xra[1]-xra[0])/dx*cosdec))
     ny=int(np.ceil((yra[1]-yra[0])/dx))
     
     avmap=np.zeros([ny,nx])
     
     # cell indices
     x=np.arange(nx)
     X=np.tile(x,(ny,1))
     y=np.arange(ny)
     Y=np.tile(y,(nx,1)).T
     
     # WCS
     w=wcs.WCS(naxis=2)
     w.wcs.crpix=[nx/2,ny/2]
     w.wcs.cdelt=np.array([-dx,dx])
     w.wcs.crval=[np.mean(xra),np.mean(yra)]
     w.wcs.ctype=['RA---TAN','DEC--TAN']
     
     # cell coords [count from zero]
     RA=np.zeros([ny,nx])
     Dec=np.zeros([ny,nx])
     for i,(x,y) in enumerate(zip(X,Y)):
          RA[i],Dec[i]=w.wcs_pix2world(np.c_[x,y],0).T
     
     # linear distance in RA/Dec - what is max error compared to haversine?
     r0,d0,r1,d1 = map(np.radians,[RA.min(),Dec.min(),RA.max(),Dec.max()])
     dlon = r1-r0
     dlat = d1-d0
     # https://en.wikipedia.org/wiki/Great-circle_distance
     grtcircledist = np.arctan2(
         np.sqrt( (np.cos(d1)*np.sin(dlon))**2 +
                  (np.cos(d0)*np.sin(d1) - np.sin(d0)*np.cos(d1)*np.cos(dlon))**2 ),
         np.sin(d0)*np.sin(d1) + np.cos(d0)*np.cos(d1)*np.cos(dlon) )
     lineardist=np.sqrt((dlon*np.cos(d0))**2+dlat**2)
     
     print("max error from linear WCS = ",2*np.abs(grtcircledist-lineardist)/(grtcircledist+lineardist))
     
     # back to the business at hand - weighted mean of nnear nearest AVs:
     for ((i,j),r),((k,l),d) in zip(np.ndenumerate(RA),np.ndenumerate(Dec)):
          avmap[i,j] = kNN_extinction(rastar,destar,avstar,eps,nnear,r,d)
           
     hdu=fits.PrimaryHDU(avmap,header=w.to_header())
     hdu.writeto(outroot+".fits",overwrite=True)
 
     pl.clf()
     ax=pl.subplot(111,projection = w)
     avimage=ax.imshow(avmap,cmap="gray",interpolation="nearest")
     ax.set_xlabel("Right Ascension (J2000)")
     ax.set_ylabel("Declination")
     pl.colorbar(ax=ax,label="A_V from UMS",mappable=avimage)
     pl.savefig(outroot+".png")
 
     return(avmap,w)
