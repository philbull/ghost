#!/usr/bin/python
"""
Analyse results of emcee MCMC chains
"""
import numpy as np
import pylab as P
import pandas as pd
import corner
from bestfit import *

CHAIN_FILE = "chain_gzrad_atten_3.dat"

symbols = {
    'extinction_tau0':       r'$\tau_0$',
    'extinction_beta':      r'$\beta_\tau$',
    'extinction_diskfac':   r'$a_{\rm disk}$',
    'fpass_alpha0':         r'$\alpha_f$',
    'fpass_beta':           r'$\beta_f$',
    'fpass_zeta':           r'$\zeta_f$',
    'ms_cen_beta0':         r'$\beta_\star$',
    'ms_cen_gamma0':        r'$\gamma_\star$',
    'ms_cen_logM1':         r'$\log_{10}M^\star_1$',
    'ms_cen_norm':          r'$A_\star$',
    'ms_cen_sigma1':        r'$\sigma_1^\star$',
    'ms_cen_sigmainf':      r'$\sigma_\infty^\star$',
    'sfr_pass_mshift':      r'$a_{\rm pass}$',
    'sfr_pass_sigma':       r'$\sigma_{\rm pass}$',
    'sfr_sfms_alpha0':      r'$\alpha_{\rm SFMS}$',
    'sfr_sfms_beta':        r'$\beta_{\rm SFMS}$',
    'sfr_sfms_sigma':       r'$\sigma_{\rm SFMS}$',
}

# Ordered list of parameter names
pnames = [
    'ms_cen_beta0',
    'ms_cen_gamma0',
    'ms_cen_logM1',
    'ms_cen_norm',
    'ms_cen_sigma1',
    'ms_cen_sigmainf',
    'fpass_alpha0',
    'fpass_beta',
    'fpass_zeta',
    'sfr_sfms_alpha0',
    'sfr_sfms_beta',
    'sfr_sfms_sigma',
    'sfr_pass_mshift',
    'sfr_pass_sigma',
    'extinction_tau0',
    'extinction_beta',
    'extinction_diskfac',
]

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

def thinned_chain(chain, nworkers, burnin, thin):
    """
    Construct thinned, multi-worker sample.
    """
    newchain = {}
    for key in chain.keys():
        tmp = []
        for wkr in range(nworkers):
            tmp.append( chain[key][wkr::nworkers][burnin:][::thin] )
        newchain[key] = np.concatenate(tmp)
    return newchain

def moving_avg(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def autocorr(x):
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]

# Load MCMC chain from file
chain1 = load_chain(CHAIN_FILE, cache=True)
chain = thinned_chain(chain1, nworkers=128, burnin=500, thin=32)


# Find maximum likelihood value
idx = np.where(chain['logl'] == np.max(chain['logl']))[0][0]
print "logL_max = %3.2f" % chain['logl'][idx]
lbls = chain.keys()
lbls.sort()
for key in lbls:
    if key not in ['logl', 'walker']:
        print "'%s': %3.7f," % (key, chain[key][idx])

"""
# Plot 1D chains
P.subplot(111)
#P.plot(chain['fpass_alpha0'])

for key in chain.keys():
    if key not in params0.keys(): continue
    ravg = pd.rolling_mean(chain[key], 50)
    if key in params0.keys():
        print "%15s: %5.4f %5.4f" % (key, ravg[-1], params0[key])
    P.plot(ravg / ravg[50], label=key )

P.legend(loc='lower right', frameon=False)
P.show()
"""

# Construct array of data and list of parameter labels
data = np.array([chain[key] for key in pnames])
lbls = [symbols[p] for p in pnames]
truths = [params_bf[key] for key in pnames]
print data.shape

# Re-process fields that should be logged
# 'sfr_pass_mshift', 'extinction_tau0'
#for pn in ['extinction_diskfac',]:
#    data[pnames.index(pn)] = np.log10(data[pnames.index(pn)])

# Corner plot with 1- and 2-sigma contours
fig = corner.corner(data.T, labels=lbls, plot_density=False, 
                    plot_datapoints=False, 
                    quantiles=(0.16, 0.84), 
                    levels=(1.-np.exp(-0.5), 1.-np.exp(-0.5*4.)), 
                    truths=truths,
                    truth_color='#E72327',
                    label_kwargs={'fontsize':24.},
                    smooth=0.65,
                    hist_kwargs={'lw': 2.},
                    contour_kwargs={'linewidths': 2.})
for ax in fig.axes:
    ax.tick_params(axis='both', which='major', labelsize=15, size=6., 
                        width=1.5, pad=3.)
    ax.tick_params(axis='both', which='minor', labelsize=15, size=4., 
                        width=1.5, pad=3.)
#    ax.xaxis.set_labelpad(50.)
#    ax.yaxis.set_labelpad(50.)

P.savefig("../draft/triangle.pdf")
P.show()

