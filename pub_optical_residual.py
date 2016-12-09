#!/usr/bin/python
"""
Plot residuals of optical magnitude fits in the SFR-Mstar plane.
"""
import numpy as np
import pylab as P
import galaxy_model as g
from bestfit import *

BAND = 'u'

# Best-fit parameters
params = params_bf

# Load Guo (2010) simulations data (z=0)
mh, mstar, sfr, mag_u, mag_g, mag_r, mag_i, mag_z = \
    np.genfromtxt('../delucia_mass/guo_colours_z0.dat', 
    delimiter=',', skip_header=1).T
mags = [mag_u, mag_g, mag_r, mag_i, mag_z]
names = ['u', 'g', 'r', 'i', 'z']

# Convert units
mh *= 1e10 / 0.73
mstar *= 1e10 / 0.73

# Plot 2D scatter plot of residuals


# Apply selection
band = names.index(BAND)
mag = mags[band]
idxs = np.where(sfr > -1.)

# Calculate residuals
mag_est = mag[idxs] - g.optical_mag(sfr[idxs], mstar[idxs], 
                                    band=BAND, z=0., params=params)

# Plot residuals
P.subplot(111)

P.scatter(mstar[idxs], sfr[idxs], c=mag_est, cmap='Spectral', lw=0., s=5., 
          vmin=-1., vmax=1., rasterized=True)

P.xscale('log')
P.yscale('log')
P.xlim((5e5, 3e12))
P.ylim((1e-7, 5e1))
cbar = P.colorbar()
cbar.set_label("$\Delta m$", fontsize=18)

# Cuts
#P.axhline(1e-7, ls='dashed', color='k', lw=1.8)
#P.axvline(1e7, ls='dashed', color='k', lw=2.)

P.xlabel('$M_\star$ $[M_\odot]$', fontsize=18)
P.ylabel(r'$\psi_{\rm SFR}$ $[M_\odot/{\rm yr}]$', fontsize=18)
    
P.tick_params(axis='both', which='major', labelsize=18, size=8., 
                    width=1.5, pad=8.)
P.tick_params(axis='both', which='minor', labelsize=18, size=5., 
                    width=1.5, pad=8.)

P.tight_layout()
P.savefig('../draft/residual_%sband.pdf' % BAND)
P.show()
