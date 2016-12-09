#!/usr/bin/python
"""
Plot pdf in the stellar mass-SFR plane.
"""
import numpy as np
import pylab as P
import galaxy_model as g
import matplotlib
from bestfit import *

COLOUR_SFMS = '#0088FF'
COLOUR_PASS = '#E72327'

# Get current best-fit model
params = params_bf

ALPHA1 = 0.5
ALPHA2 = 0.5

# Load my halo model
pkfile = "camb_pk_z0.dat"
hm = g.HaloModel(pkfile, h=0.67, om=0.32)


# 2D grid: SFR and Mstar
mstar = np.logspace(7., 12.2, 240)
sfr = np.logspace(-7., 4., 200)
n_sf_sfms, n_sf_pass = g.sfr_mstar_plane(hm, sfr, mstar, z=0., params=params)

# Bounding box for the M*-SFR plane
extent = [np.log10(np.min(mstar)), np.log10(np.max(mstar)),
          np.log10(np.min(sfr)), np.log10(np.max(sfr))]

# Load data from Guo (2010) semi-analytic mock catalogue
dl_mvir, dl_mstellar, dl_sfr, mag_u, mag_g, mag_r, mag_i, mag_z = \
    np.genfromtxt('../delucia_mass/guo_colours_z0.dat', 
    delimiter=',', skip_header=1).T
dl_mstellar *= 1e10 / 0.73 # convert to Msun
dl_mvir *= 1e10 / 0.73 # convert to Msun

# 2D histogram of Guo SAM sample, gives dN/dlogM*/dlogSFR
mm = np.logspace(np.log10(np.min(mstar)), np.log10(np.max(mstar)), 60)
ss = np.logspace(np.log10(np.min(sfr)), np.log10(np.max(sfr)), 39)
n_dl, xx, yy = np.histogram2d(np.log(dl_mstellar), np.log(dl_sfr), 
               bins=[np.log(mm), np.log(ss)])
Lbox = 62.5 / 0.73 # Mpc/h -> Mpc, box size
n_dl = n_dl.T / Lbox**3. # (dN/dlogM*/dlogSFR)/dV = dn/dlogM*/dlogSFR

# Density plots
#plt = P.imshow(np.log10(n_sf_sfms), origin='lower', cmap='Reds', extent=extent, 
#          vmin=-8., vmax=-2., aspect=1./2., interpolation='none', alpha=ALPHA1)
#plt = P.imshow(np.log10(n_sf_pass), origin='lower', cmap='Blues', extent=extent, 
#          vmin=-8., vmax=-2., aspect=1./2., interpolation='none', alpha=ALPHA1)
plt = P.imshow(np.log10(n_dl), origin='lower', cmap='YlOrBr', extent=extent, 
          vmin=-8., vmax=-2., aspect=1./2., interpolation='none', alpha=0.75)

# Plot contours from my model
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
MM, SS = np.meshgrid(np.log10(mstar), np.log10(sfr))

cplt1 = P.contour(MM, SS, np.log10(n_sf_sfms), levels=[-8., -6., -4., -3., -2.],
                  linewidths=2.5, colors=COLOUR_SFMS)
cplt2 = P.contour(MM, SS, np.log10(n_sf_pass), levels=[-8., -6., -4., -3., -2.],
                  linewidths=2., colors=COLOUR_PASS)
#P.clabel(cplt1, fmt='$10^{%d}$', fontsize=16)
#P.clabel(cplt2, fmt='$10^{%d}$', fontsize=16)


#-------------------------------------------------------------------------------
# Plot best-fit powerlaw curves from Wang et al. (2013)
f_sfr = lambda ms, alpha, beta: np.log10(10.**alpha * (ms)**beta)
P.plot(np.log10(mstar), f_sfr(mstar, -3.14, 0.37), 'k-', lw=2.4, 
       label="Wang fixed beta", dashes=[4,3])
#P.plot(logms, f_sfr(mstar, -4.65, 0.5), 'b--', lw=1.5, label="Wang by-eye fit")
#P.legend(loc='lower right', frameon=False)

#-------------------------------------------------------------------------------

P.xlim((np.min(np.log10(mstar)), np.max(np.log10(mstar))))
P.ylim((np.min(np.log10(sfr)), np.max(np.log10(sfr))))
P.xlabel(r"$\log_{10} M_\star$ $[M_\odot]$", fontsize=18)
P.ylabel(r"$\log_{10} \psi$ $[M_\odot/{\rm yr}]$", fontsize=18)
#P.colorbar()

P.gca().tick_params(axis='both', which='major', labelsize=20, size=8., 
                    width=1.5, pad=8.)
P.gca().tick_params(axis='both', which='minor', labelsize=20, size=5., 
                    width=1.5, pad=8.)

P.tight_layout()
P.savefig('../draft/sfr_mstar.pdf')
P.show()
