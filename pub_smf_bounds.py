#!/usr/bin/python
"""
Plot mean stellar mass function, with credible intervals.
"""
import numpy as np
import pylab as P
import scipy.interpolate
import galaxy_model as g
import likelihoods as like
import time
from bestfit import *

NUM_SAMPLES = 1000 #2000


COLOUR_SFMS = '#1D76C5' #'#0088FF'
COLOUR_PASS = '#E72327'

np.random.seed(15)
#CHAIN_FILE = "chain_new_gzrad_atten_burnt.dat"
CHAIN_FILE = "chain_gzrad_atten_3.dat"

# Use best-fit parameters
params = params_bf

# Load my halo model
pkfile = "camb_pk_z0.dat"
hm = g.HaloModel(pkfile, h=0.67, om=0.32)

# Start timing
tstart = time.time()

def load_chain(fname, cache=True):
    """
    Load emcee chain from a file.
    """
    # Open file and extract header
    f = open(fname, 'r')
    hdr = f.readline()[2:-1] # Trim leading hash and trailing newline
    hdr = hdr.split(' ')
    f.close()
    
    # Load data (caching if necessary)
    if cache:
        try:
            dat = np.load("%s.npy" % fname)
        except:
            dat = np.genfromtxt(fname).T
            np.save(fname, dat)
    else:
        dat = np.genfromtxt(fname).T
    
    # Repack into dictionary
    ddict = {}
    for i in range(len(hdr)):
        ddict[hdr[i]] = dat[i]
    return ddict

def moving_avg(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def sfr_sfms(mstar, alpha0, beta):
    """
    Mean SFR of the star-formation main sequence, at z=0.
    """
    sfr = 10.**alpha0 * (mstar/1e10)**beta
    return sfr


# Load SDSS-GALEX stellar mass function
mstar_sdss, sf_phi_sdss, sf_errp_sdss, sf_errm_sdss, \
            qu_phi_sdss, qu_errp_sdss, qu_errm_sdss \
                = like.load_sdss_smf(h=0.67, mstar_min=None, 
                                     convert_errors=True)
# Trim low-mass datapoints
#mstar_sdss, sf_phi_sdss, sf_errp_sdss, sf_errm_sdss, \
#qu_phi_sdss, qu_errp_sdss, qu_errm_sdss \
#  = [var[4:] for var in (mstar_sdss, sf_phi_sdss, sf_errp_sdss, sf_errm_sdss,
#                         qu_phi_sdss, qu_errp_sdss, qu_errm_sdss)]
sf_logphi_sdss = np.log10(sf_phi_sdss)
qu_logphi_sdss = np.log10(qu_phi_sdss)


# Load MCMC chain from file
dat = load_chain(CHAIN_FILE, cache=True)
#print dat.keys()

# Loop over some number of samples
#mstar = np.logspace(12., 13., 4)
mstar = np.logspace(8., 13., 60)
fpass = []; dndlogms = []
for i in np.random.randint(0, dat['logl'].size, size=NUM_SAMPLES):
    print i
    
    # Update set of params
    for key in dat.keys():
        params[key] = dat[key][i]
    
    # Calculate SMF and passive fraction for this set of params
    _dndlogms = g.stellar_mass_fn(hm, mstar, z=0., params=params)
    _fpass = g.f_passive(mstar, z=0., params=params)
    dndlogms.append(_dndlogms)
    fpass.append(_fpass)

dndlogms = np.array(dndlogms)
fpass = np.array(fpass)

# Print timing info
print "Run took %2.1f sec." % (time.time() - tstart)

# Get median and 1/2-sigma bounds at each M*
vals_sfms = []; vals_pass = []
for i in range(2):
    for j in range(mstar.size):
        # Histogram values, and estimate cdf with cumulative sum
        y = dndlogms * (1. - fpass) if i == 0 else dndlogms * fpass
        hist, x = np.histogram( np.log10(y[:,j]), bins=100, normed=True)
        #hist, x = np.histogram( y[:,j], bins=100, normed=True)
        xc = 10.**(0.5 * (x[1:] + x[:-1]))
        #xc = (0.5 * (x[1:] + x[:-1]))
        cumsum = np.cumsum(hist)
        cumsum /= cumsum[-1] # Normalise
        
        # Build interpolation function and evaluate at median and 1/2-sigma
        cdf = scipy.interpolate.interp1d(cumsum, xc, kind='linear', 
                                         bounds_error=False)
        if i == 0:
            vals_sfms.append( [cdf(0.5), cdf(0.16), cdf(0.84), 
                               cdf(0.025), cdf(0.975)] )
        else:
            vals_pass.append( [cdf(0.5), cdf(0.16), cdf(0.84), 
                               cdf(0.025), cdf(0.975)] )
vals_sfms = np.array(vals_sfms).T
vals_pass = np.array(vals_pass).T


# Plot density contours
P.subplot(111)

# SFMS, median and 1/2-sigma
P.fill_between(mstar, vals_sfms[3], vals_sfms[4], color=COLOUR_SFMS, alpha=0.4)
P.fill_between(mstar, vals_sfms[1], vals_sfms[2], color=COLOUR_SFMS, alpha=0.4)
P.plot(mstar, vals_sfms[0], 'k-', lw=1.8) # Median

# Passive, median and 1/2-sigma
P.fill_between(mstar, vals_pass[3], vals_pass[4], color=COLOUR_PASS, alpha=0.4)
P.fill_between(mstar, vals_pass[1], vals_pass[2], color=COLOUR_PASS, alpha=0.4)
P.plot(mstar, vals_pass[0], 'k-', lw=1.8) # Median


# Plot best-fit powerlaw curves from Wang et al. (2013)
#f_sfr = lambda ms, alpha, beta: 10.**alpha * (ms)**beta
#P.plot(mstar, f_sfr(mstar, -3.14, 0.37), 'k-', lw=2.4, 
#       label="Wang fixed beta", dashes=[4,3])

P.errorbar(mstar_sdss, sf_phi_sdss, yerr=[sf_errm_sdss, sf_errp_sdss], 
           color=COLOUR_SFMS, marker='.', ls='none', capsize=4., 
           elinewidth=1.5, mew=1.5)

P.errorbar(mstar_sdss, qu_phi_sdss, yerr=[qu_errm_sdss, qu_errp_sdss], 
           color=COLOUR_PASS, marker='.', ls='none', capsize=4., 
           elinewidth=1.5, mew=1.5)

P.xlabel(r"$M_\star$ $[M_\odot]$", fontsize=18)
P.ylabel(r"$dn/d\log M_\star$ $[{\rm Mpc}^{-3}]$", fontsize=18)

P.gca().tick_params(axis='both', which='major', labelsize=20, size=8., 
                    width=1.5, pad=8.)
P.gca().tick_params(axis='both', which='minor', labelsize=20, size=5., 
                    width=1.5, pad=8.)

P.xscale('log')
P.yscale('log')

P.xlim((5e8, 3e12))
P.ylim((3e-9, 4e-2))

P.tight_layout()
P.savefig('../draft/smf_bounds_both.pdf')
P.show()
