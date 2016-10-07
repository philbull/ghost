#!/usr/bin/python
"""
Likelihood functions for various data.
"""
import numpy as np
import pylab as P
from scipy.special import erf

def load_mauch_lf(fname="../lumfns/lumfunc_6dfgs.dat", h=0.7, starburst_corr=False):
    """
    Load 6dFGS star-forming radio galaxy data from Mauch & Sadler 
    (astro-ph/0612018). (z_median = 0.035, assumed h=0.7, omega_m=0.3)
    """
    h_ms = 0.7
    
    # Load data from file
    log10Lum, log10Phi, errp, errm = np.genfromtxt(fname).T
    fac = 2.5/np.log(10.) # Convert d/d(mag) to 1/d(logL)
    
    # Convert luminosity units
    L = 10.**log10Lum * 1e7 # erg/s/Hz
    
    # Approximate correction to luminosity at low-z for different cosmologies, 
    # assuming d_L ~ c/H_0.
    L *= (h / h_ms)**2.
    
    # Approximate correction to the volume, taking into account different H_0
    Vfac = (h / h_ms)**3. # 
    
    # Calculate Phi,and rescale log-errors 
    Phi = fac * Vfac * 10.**log10Phi
    # Unit rescaling *should not* be applied to log-errors
    # (the rescaling is already included in the units of Phi)
    #errp += np.log10(fac * Vfac)
    #errm += np.log10(fac * Vfac)
    
    # Apply correction to remove "starburst" galaxies assumed to be 
    # contaminating the LF. Based on fit to the ratio of "normal" star-forming  
    # to "total" (normal+starburst) galaxy radio LFs (Yun, Reddy & Condon 2001).
    if starburst_corr:
        Lref = 6e29 # Fitting parameter
        corr = 0.5 * ( 1. - erf(L/Lref - 1.) )
        Phi *= corr
        # FIXME: Does not correct the errorbars
    
    return L, Phi, errp, errm


def load_gama_lf(band, froot="../lumfns/lf%s_z0_driver12.data", h=0.7):
    """
    Load GAMA optical luminosity functions from Driver et al. (2012).
    (assumes omega_m = 0.27, and in h units)
    """
    # Load GAMA binned luminosity fun. for a given band
    gama_mag, gama_n, gama_err, gama_ngal = np.genfromtxt(froot % band).T
    gama_mag += 5.*np.log10(h)
    gama_n *= h**3. # Convert (Mpc/h)^-3 -> (Mpc)^-3
    gama_err *= h**3.
    # FIXME: No conversion for different Omega_M
    
    # Remove unconstrained bins
    idxs = np.where(gama_err > 0.)
    
    return gama_mag[idxs], gama_n[idxs], gama_err[idxs]

def load_sdss_smf(froot="../lumfns/moustakas_smf.dat", h=0.7, 
                  convert_errors=False, mstar_min=None):
    """
    Load SDSS-GALEX stellar mass functions (z = 0.01 - 0.2) from Moustakas et 
    al. 2013 [1301.1688].
    """
    # Stellar mass function from Table 3, with columns for all, star-forming 
    # only, and quiescent-only
    logms, \
    all_logphi, all_errp, all_errm, all_sig, all_N, \
    sf_logphi, sf_errp, sf_errm, sf_sig, sf_N, \
    qu_logphi, qu_errp, qu_errm, qu_sig, qu_N = np.genfromtxt(froot).T
    
    # Convert units
    # Stellar mass in units of log_10(h_70^-2 Msun), where 
    # h_70 = H_0 / (70 km/s/Mpc). Convert to Msun units.
    logms += -np.log10((h / 0.7)**2.)
    ms = 10.**logms
    
    # Stellar mass function, in units of log_10(h_70^3 Mpc^-3 dex^-1)
    # Convert to log_10 (Mpc^-3), i.e. dn/dlogM/dV
    all_logphi += np.log10((h / 0.7)**3. / np.log(10.))
    sf_logphi += np.log10((h / 0.7)**3. / np.log(10.))
    qu_logphi += np.log10((h / 0.7)**3. / np.log(10.))
    
    # Convert from log-space
    all_phi = 10.**all_logphi
    sf_phi = 10.**sf_logphi
    qu_phi = 10.**qu_logphi
    
    # If requested, convert errors from log space
    if convert_errors:
        all_errp = all_phi * (10.**all_errp - 1.)
        sf_errp = sf_phi * (10.**sf_errp - 1.)
        qu_errp = qu_phi * (10.**qu_errp - 1.)
        
        all_errm = all_phi * (1. - 10.**all_errm)
        sf_errm = sf_phi * (1. - 10.**sf_errm)
        qu_errm = qu_phi * (1. - 10.**qu_errm)
    
    # Apply a cut (minimum) in mstar
    if mstar_min is not None:
        idxs = np.where(ms >= mstar_min)
        ms, sf_phi, sf_errp, sf_errm, qu_phi, qu_errp, qu_errm = \
            [f[idxs] 
             for f in [ms, sf_phi, sf_errp, sf_errm, qu_phi, qu_errp, qu_errm]]
    return ms, sf_phi, sf_errp, sf_errm, qu_phi, qu_errp, qu_errm
    
