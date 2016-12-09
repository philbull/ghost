#!/usr/bin/python
"""
Plot passive fraction, with credible intervals.
"""
import numpy as np
import pylab as P
import scipy.interpolate
import galaxy_model as g
import likelihoods as like

COLOUR_SFMS = '#0088FF'
COLOUR_PASS = '#E72327'

np.random.seed(15)
#CHAIN_FILE = "chain_new_gzrad_atten_burnt.dat"
CHAIN_FILE = "chain_gzrad_atten_3.dat"

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

def f_passive(mstar, alpha0, beta, zeta):
    """
    Fraction of passive galaxies at a given stellar mass and redshift.
    """
    c = 0.5 * (1. + np.tanh(zeta))
    return c + (1. - c) / ( 1. + (mstar / (10.**alpha0))**beta )


# Load MCMC chain from file
dat = load_chain(CHAIN_FILE, cache=True)
#print dat.keys()

mstar = np.logspace(np.log10(5e8), np.log10(2e12), 100) # 100
print dat['fpass_alpha0'].size
alpha0 = dat['fpass_alpha0'][10000:]
beta = dat['fpass_beta'][10000:]
zeta = dat['fpass_zeta'][10000:]

# Load SDSS-GALEX stellar mass function, and estimate f_pass and errorbars
mstar_sdss, sf_phi_sdss, sf_errp_sdss, sf_errm_sdss, \
            qu_phi_sdss, qu_errp_sdss, qu_errm_sdss \
                             = like.load_sdss_smf(h=0.67)
s = sf_phi_sdss
p = qu_phi_sdss
fpass_sdss = p / (s + p)
# Gaussian error propagation
err_sdss = np.sqrt(  qu_errp_sdss**2. * (p/(s+p))**2. 
                   + sf_errp_sdss**2. * ((s+p) * (1. - 1./(s+p)))**2. )


# Calculate f_passive on a grid for all values in the chain
MSTAR, ALPHA0 = np.meshgrid(mstar, alpha0)
MSTAR, BETA = np.meshgrid(mstar, beta)
MSTAR, ZETA = np.meshgrid(mstar, zeta)
fpass = f_passive(MSTAR, ALPHA0, BETA, ZETA)

# Get median and 1/2-sigma bounds at each M*
vals = []
for i in range(mstar.size):
    # Histogram values, and estimate cdf with cumulative sum
    hist, x = np.histogram(fpass[:,i], bins=100, normed=True)
    xc = 0.5 * (x[1:] + x[:-1])
    cumsum = np.cumsum(hist)
    cumsum /= cumsum[-1] # Normalise
    
    # Add leading zero
    xc = np.concatenate(([xc[0] - (xc[1]-xc[0]),], xc))
    cumsum = np.concatenate(([0.,], cumsum))
    
    # Build interpolation function and evaluate at median and 1/2-sigma
    cdf = scipy.interpolate.interp1d(cumsum, xc, kind='linear', bounds_error=True)
    vals.append([cdf(0.5), cdf(0.16), cdf(0.84), cdf(0.025), cdf(0.975)])
vals = np.array(vals).T

# Plot density contours
P.subplot(111)
P.fill_between(mstar, vals[3], vals[4], color=COLOUR_PASS, alpha=0.4) # 2-sigma
P.fill_between(mstar, vals[1], vals[2], color=COLOUR_PASS, alpha=0.4) # 1-sigma
P.plot(mstar, vals[0], 'k-', lw=1.8) # Median

# Plot SDSS/GALEX data points
P.errorbar(mstar_sdss, fpass_sdss, yerr=err_sdss, ms=7., color='k', 
           ls='none', marker='.', capsize=4., elinewidth=1.5, mew=1.5)

# Assumed f_pass function from Behroozi et al. [1207.6105] Eq. 8
f_pass = lambda ms: 1. / ( (ms/10.**(10.2))**-1.3 + 1. )
P.plot(mstar, f_pass(mstar), color='#3841F2', lw=2.8, dashes=[4,3])
#'#2382C2'

P.xlabel(r"$M_\star$ $[M_\odot]$", fontsize=18)
P.ylabel(r"$f_{\rm pass}$", fontsize=18)

P.gca().tick_params(axis='both', which='major', labelsize=20, size=8., 
                    width=1.5, pad=8.)
P.gca().tick_params(axis='both', which='minor', labelsize=20, size=5., 
                    width=1.5, pad=8.)

P.xscale('log')
#P.yscale('log')
P.ylim((0., 1.05))
P.xlim((5e8, 2e12))

P.tight_layout()
P.savefig('../draft/fpass_bounds.pdf')
P.show()
