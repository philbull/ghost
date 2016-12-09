#!/usr/bin/python
"""
Plot scaling of dust optical depth with wavelength and stellar mass.
"""
import numpy as np
import pylab as P
import galaxy_model as g

params = {}
params['extinction_amp'] = 0.5
params['extinction_beta'] = 0.4 #1.78825
params['extinction_diskfac'] = 0.3 #2.19008
params['extinction_alpha'] = -1. #-7.57886


def tau_extinction(sintheta, mstar, band, z, params):
    """
    Dust extinction optical depth as a function of inclination angle.
    """
    # Extinction parameters
    A = params['extinction_amp']
    beta = params['extinction_beta']
    kappa = params['extinction_diskfac']
    alpha = params['extinction_alpha']
    
    # Get band wavelength (assumed to be a number if a band ID is not specified
    if band in g.band_wavelength.keys():
        l = g.band_wavelength[band]
    else:
        l = float(band)
    
    # Calculate optical depth and return
    return 1.086 * A * (mstar / 1e10)**beta * (1. + kappa*sintheta) \
             * (l / 5000.)**alpha

def Anu_pannella(mstar):
    # Attenuation at 1500 Angstrom, from Pannella et al. (2009)
    return 4.07 * np.log10(mstar) - 39.32


# Scaling with stellar mass
P.subplot(211)

mstar = np.logspace(9., 12., 200)
P.plot(mstar, tau_extinction(0., mstar, 1500., 0., params), 'b-', lw=1.8)
P.plot(mstar, tau_extinction(1., mstar, 1500., 0., params), 'b-', lw=1.8)

P.plot(mstar, Anu_pannella(mstar), 'r-', lw=1.5)

P.xscale('log')
#P.yscale('log')

P.xlim((1e10, 10.**11.2))
P.ylim((0., 7.5))
P.xlabel("M*")
P.ylabel(r"$A_\nu$")

P.grid()


# Scaling with wavelength
P.subplot(211)
l = np.linspace(1000., 10000., 200)

# Prevot 1984, see http://webast.ast.obs-mip.fr/hyperz/hyperz_manual1/node10.html
k_lambda = [4.20, 3.70, 3.30, 3.00, 2.70, 2.80, 2.90, 2.30, 1.97, 1.69, 1.58, 
            1.45, 1.32, 1.13, 1.00, 0.46, 0.74]
lam = [1000., 1110., 1250., 1430., 1670., 2000., 2220., 2500., 2850., 3330., 
       3650., 4000., 4400., 5000., 5530., 9000., 6700.]

P.plot(lam, k_lambda, 'k-', marker='.', lw=1.8)
P.plot(l, tau_extinction(0., 1e10, l, 0., params), 'r-', lw=1.8)
P.plot(l, tau_extinction(1., 1e10, l, 0., params), 'r-', lw=1.8)

P.xscale('log')

#P.xlim((1e10, 10.**11.2))
#P.ylim((0., 7.5))
P.xlabel("$\lambda$")
P.ylabel(r"$A_\nu$")

P.tight_layout()
P.show()
