import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cmasher as cmr

filtdir = '/Users/toneill/N159/hst_filts/'


from astropy.io import ascii


f125 = ascii.read(filtdir+'HST_WFC3_IR.F125W.dat').to_pandas()
f110 = ascii.read(filtdir+'HST_WFC3_IR.F110W.dat').to_pandas()
f160 = ascii.read(filtdir+'HST_WFC3_IR.F160W.dat').to_pandas()

f555 = ascii.read(filtdir+'HST_ACS_WFC.F555W.dat').to_pandas()
f814 = ascii.read(filtdir+'HST_ACS_WFC.F814W.dat').to_pandas()
f775 = ascii.read(filtdir+'HST_ACS_WFC.F775W.dat').to_pandas()

lprops = {'lw':2}
fillprops = {'alpha':0.45}

###########################

#cmap_use = cmr.get_sub_cmap(cmr.infinity, 0.15, 1)  # cmr.ember, 0.2, 0.8)

#ax.set_prop_cycle('color',cmap_use(np.linspace(0,1,6)))#plt.cm.jet(np.linspace(0,1,6)))

plt.figure(figsize=(10,8))
ax = plt.subplot(111)
ax.plot(f555['col1'],f555['col2'],label='F555W',**lprops,color='mediumblue')
ax.fill_between(f555['col1'],np.repeat(0,len(f555)),f555['col2'],**fillprops,color='mediumblue')

plt.plot(f775['col1'],f775['col2'],label='F775W',color='royalblue',ls='--',lw=2)
plt.fill_between(f775['col1'],np.repeat(0,len(f775)),f775['col2'],alpha=0.3,color='royalblue',zorder=4)

plt.plot(f814['col1'],f814['col2'],label='F814W',color='mediumseagreen',zorder=3,**lprops)
plt.fill_between(f814['col1'],np.repeat(0,len(f814)),f814['col2'],alpha=0.4,color='mediumseagreen',zorder=3)

plt.plot(f110['col1'],f110['col2'],label='F110W',color='gold',zorder=2,ls='--',lw=3)
plt.fill_between(f110['col1'],np.repeat(0,len(f110)),f110['col2'],alpha=0.3,color='gold',zorder=2)

plt.plot(f125['col1'],f125['col2'],label='F125W',color='darkorange',zorder=0,**lprops)
plt.fill_between(f125['col1'],np.repeat(0,len(f125)),f125['col2'],alpha=0.3,zorder=0,color='darkorange')

plt.plot(f160['col1'],f160['col2'],label='F160W',zorder=1,color='firebrick',**lprops)
plt.fill_between(f160['col1'],np.repeat(0,len(f160)),f160['col2'],**fillprops,color='firebrick')

plt.legend(loc='upper left',fontsize=16)
plt.ylim(0,0.65)
plt.xlabel('$\\lambda$ ($\\AA$)',fontsize=20)
plt.ylabel('Transmission',fontsize=20)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(direction='in', which='both', labelsize=12)
plt.gca().set_xscale('log')
plt.tight_layout()

#plt.savefig('hst_bands.png',dpi=300)


##################################################

havecol = 'gray'
wantcol = 'r'

plt.figure(figsize=(10,8))
ax = plt.subplot(111)

plt.fill_between(f555['col1'],np.repeat(0,len(f555)),f555['col2'],**fillprops,color=havecol,label='Have in Existing Training Set')
plt.fill_between(f775['col1'],np.repeat(0,len(f775)),f775['col2'],alpha=0.3,color=havecol,zorder=4)
plt.fill_between(f110['col1'],np.repeat(0,len(f110)),f110['col2'],alpha=0.3,color=havecol,zorder=2)
plt.fill_between(f160['col1'],np.repeat(0,len(f160)),f160['col2'],**fillprops,color=havecol)


plt.plot(f555['col1'],f555['col2'],**lprops,color=wantcol,label='Desired for Training Set')
plt.plot(f814['col1'],f814['col2'],color=wantcol,zorder=3,**lprops)
plt.plot(f125['col1'],f125['col2'],color=wantcol,zorder=0,**lprops)
plt.plot(f160['col1'],f160['col2'],zorder=1,color=wantcol,**lprops)

plt.legend(loc='upper left',fontsize=16)
plt.ylim(0,0.65)
plt.xlabel('$\\lambda$ ($\\AA$)',fontsize=20)
plt.ylabel('Transmission',fontsize=20)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.tick_params(direction='in', which='both', labelsize=12)
#plt.gca().set_xscale('log')
plt.tight_layout()






