#!/usr/bin/python
"""
Plot mean SFMS, with credible intervals.
"""
import numpy as np
import pylab as P
import scipy.interpolate
import galaxy_model as g

COLOUR_SFMS = '#1D76C5' #'#0088FF'
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

def sfr_sfms(mstar, alpha0, beta):
    """
    Mean SFR of the star-formation main sequence, at z=0.
    """
    sfr = 10.**alpha0 * (mstar/1e10)**beta
    return sfr

# Load MCMC chain from file
dat = load_chain(CHAIN_FILE, cache=True)
#print dat.keys()

mstar = np.logspace(8., 12., 10)
alpha0 = dat['sfr_sfms_alpha0']
beta = dat['sfr_sfms_beta']

# Calculate SFMS relation on a grid for all values in the chain
MSTAR, ALPHA0 = np.meshgrid(mstar, alpha0)
MSTAR, BETA = np.meshgrid(mstar, beta)
sfr = sfr_sfms(MSTAR, ALPHA0, BETA)

# Get median and 1/2-sigma bounds at each M*
vals = []
for i in range(mstar.size):
    # Histogram values, and estimate cdf with cumulative sum
    hist, x = np.histogram(sfr[:,i], bins=100, normed=True)
    xc = 0.5 * (x[1:] + x[:-1])
    cumsum = np.cumsum(hist)
    cumsum /= cumsum[-1] # Normalise
    
    # Build interpolation function and evaluate at median and 1/2-sigma
    cdf = scipy.interpolate.interp1d(cumsum, xc, kind='linear')
    vals.append([cdf(0.5), cdf(0.16), cdf(0.84), cdf(0.025), cdf(0.975)])
vals = np.array(vals).T

# Plot density contours
P.subplot(111)

# Plot best-fit powerlaw curves from Wang et al. (2013)
f_sfr = lambda ms, alpha, beta: 10.**alpha * (ms)**beta

P.fill_between(mstar,
               f_sfr(mstar, -3.14 - 2.*0.46, 0.37),
               f_sfr(mstar, -3.14 + 2.*0.46, 0.37), 
               color='g', alpha=0.1)
P.fill_between(mstar,
               f_sfr(mstar, -3.14 - 0.46, 0.37),
               f_sfr(mstar, -3.14 + 0.46, 0.37), 
               color='g', alpha=0.1)
P.plot(mstar, f_sfr(mstar, -3.14, 0.37), color='#41813E', lw=2.4, 
       label="Wang fixed beta", dashes=[4,3])



speagle_sfr = lambda mstar: 10.**( (0.84-0.026*13.7)*np.log10(mstar*0.69) -(6.51-0.11*13.7) )

P.plot(mstar, speagle_sfr(mstar), 'b-', lw=1.8)


# Plot contours for MCMC samples
P.fill_between(mstar, vals[3], vals[4], color=COLOUR_SFMS, alpha=0.4) # 2-sigma
P.fill_between(mstar, vals[1], vals[2], color=COLOUR_SFMS, alpha=0.4) # 1-sigma
P.plot(mstar, vals[0], 'k-', lw=1.8) # Median

# Plot approx. stellar mass completeness limit from Wang (Table 1)
#P.axvline(4e9, color='k', alpha=0.5, lw=1.8, zorder=-100, dashes=[3,2])
P.axvline(8e9, color='k', alpha=0.5, lw=1.8, zorder=-100, dashes=[3,2])
P.fill_between([4e9, 8e9], 1e-3, 1e3, linewidths=0., color='k', alpha=0.2, zorder=-100)
P.fill_between([1e8, 4e9], 1e-3, 1e3, linewidths=0., color='k', alpha=0.1, zorder=-100)


P.xlabel(r"$M_\star$ $[M_\odot]$", fontsize=18)
P.ylabel(r"$\psi_{\rm SFR}$ $[M_\odot/{\rm yr}]$", fontsize=18)

P.gca().tick_params(axis='both', which='major', labelsize=20, size=8., 
                    width=1.5, pad=8.)
P.gca().tick_params(axis='both', which='minor', labelsize=20, size=5., 
                    width=1.5, pad=8.)

P.xscale('log')
P.yscale('log')

P.tight_layout()
#P.savefig('../draft/sfr_mstar_bounds.pdf')
P.show()
