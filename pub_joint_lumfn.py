#!/usr/bin/python
"""
Plot joint luminosity function for SFR tracer + dust-atten. optical magnitudes.
"""
import numpy as np
import pylab as P
import matplotlib
import matplotlib.gridspec as gridspec
import galaxy_model as g
import likelihoods as like
from bestfit import *
import time

BAND = 'r'

bands = ['u', 'g', 'r', 'i', 'z']
colours = ['#DA49DA', '#00a5ff', '#ff8200', '#F34334', '#9D2020'] # Freq. -> colour, but more contrasty

# Get best-fit parameters
params = params_bf

# Load my halo model
pkfile = "camb_pk_z0.dat"
hm = g.HaloModel(pkfile, h=0.67, om=0.32)

# Timing info
tstart = time.time()

#-------------------------------------------------------------------------------
# Set-up gridspec
P.figure(figsize=(16,4))
gs = gridspec.GridSpec(1,5)
gs.update(wspace=0.0, hspace=0.0)

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'

# Evaluate joint lum. fn. on a grid of SFR and optical mag.
sfr = np.logspace(-4., 3., 101)
mag = np.linspace(-26.5, -11., 100)

# Loop over bands
for b, c in zip(bands, colours):
    print "%s-band" % b
    ax1 = P.subplot(gs[bands.index(b)])
    
    # Evaluate joint lum. fn. for attenuated and unattenuated opt. mags.
    n_sfms_atten, n_pass_atten = g.joint_lumfn_sfr_optical(hm, sfr, mag, b, 
                                              z=0., params=params, atten=True)
    n_sfms, n_pass = g.joint_lumfn_sfr_optical(hm, sfr, mag, b, z=0., 
                                           params=params, atten=False)
    
    # Combine into a single luminosity function
    n_atten = n_sfms_atten #+ n_pass_atten
    n_noatten = n_sfms #+ n_pass

    # Convert from d/dSFR to d/dlogSFR
    SFR, MAG = np.meshgrid(sfr, mag)
    n_atten *= SFR.T
    n_noatten *= SFR.T
    
    # Plot results (contour plots)
    cplt1 = ax1.contour(MAG, SFR, np.log10(n_atten).T, 
                      levels=[-8., -6., -4., -3., -2.],
                      linewidths=2.5, colors=c, alpha=0.5, linestyles='dashed')

    cplt2 = ax1.contour(MAG, SFR, np.log10(n_noatten).T, 
                      levels=[-8., -6., -4., -3., -2.],
                      linewidths=2.5, colors=c)
    
    #ax1.clabel(cplt1, fmt='$10^{%d}$', fontsize=16)
    
    # Plot settings
    #ax1.set_xlim((-26.8, -11.3))
    ax1.invert_xaxis()
    ax1.set_yscale('log')
    ax1.set_ylim((1e-4, 5e3))

    ax1.tick_params(axis='both', which='major', labelsize=18, size=8., 
                        width=1.25, pad=8.)
    ax1.tick_params(axis='both', which='minor', labelsize=18, size=5., 
                        width=1.25, pad=8.)
    
    ax1.xaxis.set_major_locator( matplotlib.ticker.MultipleLocator(4.) )
    ax1.xaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(1.) )
    
    # Add band label to each upper subplot
    ax1.annotate("%s" % b, xy=(0.82, 0.08), xycoords='axes fraction', fontsize=24.)
    
    if b != 'z':
        ax1.tick_params(axis='y', which='both', right='off')
    
    if b == 'u':
        # Axis labels only on first panel
        ax1.set_ylabel(r'$\psi_{\rm SFR}$ $[M_\odot {\rm yr}^{-1}]$', 
                       fontsize=18., labelpad=10.)
        ax1.set_xlabel(r'$m_\nu$', fontsize=18.)
    else:
        # Remove y-axis tick labels where not needed
        ax1.tick_params(axis='y', which='major', labelleft='off')

print "Run took %3.1f sec." % (time.time() - tstart)

P.tight_layout()
P.savefig('../draft/joint_lumfn.pdf')
P.show()
