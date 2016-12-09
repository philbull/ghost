#!/usr/bin/python
"""
Plot luminosity function for all optical bands, vs. the GAMA data.
"""
import numpy as np
import pylab as P
import galaxy_model as g
import likelihoods as like
import copy, time
import matplotlib.ticker
import matplotlib.gridspec as gridspec
import scipy.interpolate
from bestfit import *

# Get current best-fit model
params = params_bf

#-------------------------------------------------------------------------------
# Calculate luminosity function from my model

# Load my halo model
pkfile = "camb_pk_z0.dat"
hm = g.HaloModel(pkfile, h=0.67, om=0.32)

# Number density as a function of optical magnitude
mag = np.linspace(-26.8, -11.3, 60) #100)
#obsmag = np.linspace(-24., -13., 10)

# Bands and colour-coding
bands = ['u', 'g', 'r', 'i', 'z']
colours = ['b', 'g', 'r', 'y', 'm']
colours = ['#e0e7ff', '#fff5f5', '#ffe1c6', '#ffd1a3', '#ffbb78'] # Blackbody peak colours
colours = ['#610061', '#00a5ff', '#ff8200', '#a50000', '#610000'] # Freq. -> colour conversion
colours = ['#DA49DA', '#00a5ff', '#ff8200', '#F34334', '#9D2020'] # Freq. -> colour, but more contrasty

#-------------------------------------------------------------------------------
# Set-up gridspec
P.figure(figsize=(16,6))
gs = gridspec.GridSpec(2,5)
gs.update(wspace=0.0, hspace=0.0)

#-------------------------------------------------------------------------------
# Plot optical luminosity function as a fn. of magnitude

for i in range(len(bands)):
    b = bands[i]
    col = colours[i]
    
    #if b in ['u', 'g', 'r', 'i']: continue
    
    print "%s-band" % b
    ax1 = P.subplot(gs[i])
    ax2 = P.subplot(gs[i+5])
    
    # Load GAMA band binned luminosity fun. for this band
    hh = 0.67
    gama_mag, gama_n, gama_err, gama_ngal = \
                             np.genfromtxt("../lumfns/lf%s_z0_driver12.data" % b).T
    gama_mag += 5.*np.log10(hh)
    gama_n *= hh**3. # Convert (Mpc/h)^-3 -> (Mpc)^-3
    gama_err *= hh**3.
    ax1.errorbar(gama_mag, gama_n, yerr=gama_err, marker='.', color=col, 
                 ls='none', capsize=4., elinewidth=1.5, mew=1.5)
    
    # Calculate number density as a function of magnitude
    dndmag_sfms, dndmag_pass = \
        g.optical_mag_fn_atten(hm, mag, band=b, z=0., params=params)
        #g.optical_mag_fn_dust( hm, gama_mag, mag, band=b, z=0.,
        #                       params=params, include_intrinsic=True)

    # Results from my calculation
    ax1.plot(mag, dndmag_sfms + dndmag_pass, color=col, lw=1.8) 
    #       label="%s-band" % b)
    ax1.plot(mag, dndmag_sfms, color=col, lw=1.8, ls='dashed')
    ax1.plot(mag, dndmag_pass, color=col, lw=1.8, dashes=[2,2])
    
    # Add band label to each upper subplot
    ax1.annotate("%s" % b, xy=(0.75, 0.82), xycoords='axes fraction', fontsize=24.)
    
    # Residuals (obtained by interpolating in (y-)log space)
    logdndmag_interp = scipy.interpolate.interp1d(mag, 
                               np.log(dndmag_sfms + dndmag_pass), kind='linear',
                               bounds_error=False)
    gama_n_pred = np.exp(logdndmag_interp(gama_mag))
    fracdev = (gama_n / gama_n_pred) - 1.
    fracdev[np.where(fracdev == -1.)] = np.nan # Discard zero values
    ax2.errorbar(gama_mag, fracdev,
                 yerr=gama_err / gama_n_pred, marker='.', color=col, ls='none', 
                 capsize=4., elinewidth=1.5, mew=1.5)
    ax2.axhline(0., color='k', ls='dashed', lw=1.5)
    
    # Save data to file
    #np.savetxt("model_lumfn_%s.dat" % b, 
    #           np.column_stack((mag, dndmag_sfms + dndmag_pass)))
    
    # Plot settings
    for ax in [ax1, ax2]:        
        ax.set_xlim((-26.8, -11.3))
        ax.invert_xaxis()

        ax.tick_params(axis='both', which='major', labelsize=18, size=8., 
                            width=1.25, pad=8.)
        ax.tick_params(axis='both', which='minor', labelsize=18, size=5., 
                            width=1.25, pad=8.)
        
        ax.xaxis.set_major_locator( matplotlib.ticker.MultipleLocator(4.) )
        ax.xaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(1.) )
        if i != 0:
            ax.tick_params(axis='y', which='major', labelleft='off')
    
    # y axis limits
    ax1.set_yscale('log')
    ax1.set_ylim((1e-7, 1e-1))
    #ax2.set_ylim((-1., 0.9))
    ax2.set_ylim((-1., 1.3))
    
    ax1.tick_params(axis='x', which='major', labelbottom='off')
    ax2.yaxis.set_major_locator( matplotlib.ticker.MultipleLocator(0.5) )
    ax2.yaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(0.1) )
    
    # Axis labels
    if i == 0:
        ax1.set_ylabel(r'$dn/dM_{\rm obs}$ $[{\rm Mpc}^{-3}\, {\rm mag}^{-1}]$', fontsize=18., labelpad=10.)
        ax2.set_ylabel(r'${\rm Frac.}\, {\rm diff.}$', fontsize=18., labelpad=10.)
        ax2.set_xlabel(r'$m_{\rm obs}$', fontsize=18.)

P.gcf().subplots_adjust(bottom=0.14, left=0.08, right=0.98)
P.savefig("../draft/lumfn_optical.pdf")
#P.tight_layout()
P.show()
