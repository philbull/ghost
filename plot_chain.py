#!/usr/bin/python
"""
Plot emcee MCMC chain.
"""
import numpy as np
import pylab as P
import corner
import emcee

#CHAIN_FILE = "chain_new_gzrad_atten.dat"
#CHAIN_FILE = "chain_new_gzrad_smf_atten.dat"
#CHAIN_FILE = "chain_new_gzrad_atten_burnt.dat"
#CHAIN_FILE = "chain_atten_z.dat"
#CHAIN_FILE = "chain_gzrad_atten_1.dat"
#CHAIN_FILE = "chain_gzrad_atten_2.dat"
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

def autocorr(x):
    x = x - np.mean(x)
    result = np.correlate(x, x, mode='full')
    return result[result.size/2:]

# Load MCMC chain from file
dat = load_chain(CHAIN_FILE, cache=True)
#print dat.keys()

"""
# Plot corner plot
paramnames = ['fpass_zeta', 'fpass_beta', 'sfr_sfms_alpha0', 'sfr_pass_sigma', 'fpass_alpha0', 'extinction_amp', 'extinction_beta', 'sfr_sfms_beta', 'ms_cen_logM1', 'sfr_pass_mshift', 'extinction_diskfac']
cols = []
lbls = []
for k in dat.keys():
    if k in paramnames:
        y = dat[k]
        if k in ['sfr_pass_mshift', 'sfr_pass_sigma']:
            y = np.log10(y)
        cols.append(y)
        lbls.append(k)
cols = np.array(cols).T

corner.corner(cols.T[:].T, labels=lbls[:])
P.show()
exit()
"""

"""
dat['logl'][np.where(np.isinf(dat['logl']))] = -2e5

for j in range(3):
    for skip in range(1,50):
        y = dat['sfr_sfms_alpha0'][j::128][100:]
        P.plot(skip, emcee.autocorr.integrated_time(y[::skip]), 'r.')
        #P.plot(skip, emcee.autocorr.integrated_time(y[1::skip]), 'b.')
#P.yscale('log')
"""

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


dat1 = thinned_chain(dat, nworkers=128, burnin=500, thin=35)
print dat1['logl'].shape

#-------------------------------------------------------------------------------
# Save thinned chain
names = dat1.keys()
names.sort()
data = []
for k in names:
    data.append(dat1[k])

np.savetxt("processed_chain_20161028.dat", np.column_stack(data), header=" ".join(names))

#-------------------------------------------------------------------------------

for t in np.arange(1, 100):
    dat1 = thinned_chain(dat, nworkers=128, burnin=500, thin=t)
    #mean = np.mean(dat1['sfr_sfms_alpha0'])
    #std = np.std(dat1['sfr_sfms_alpha0'])
    act = emcee.autocorr.integrated_time(dat1['sfr_sfms_beta'])
    P.plot(t, act, 'r.')
    #P.plot(t, std, 'b.')
#print emcee.autocorr.integrated_time(dat1['sfr_sfms_alpha0']), dat1['logl'].size

P.show()
exit()



# Plot parameter chains, normalised to last value
P.subplot(111)
#P.plot(dat['logl'])
#P.plot(np.abs(moving_avg(dat['logl'], 128)))
#P.plot(np.abs(moving_avg(dat['sfr_sfms_alpha0'], 128)))

#P.plot(np.abs(moving_avg(dat['sfr_sfms_beta'], 128))[50000:])
for i in range(2):
    y = dat['sfr_sfms_alpha0'][i::128][500:][::50]
    acf = autocorr(y)
    P.plot( acf )


P.axhline(0., ls='dashed', color='k')

#P.yscale('log')
#P.ylim((56., 60.))
"""
for k in dat.keys():
    if k not in ['logl', 'walker']:
        P.plot(moving_avg(dat[k]/dat[k][-1], 128),
               label=k)
P.legend(loc='upper right', frameon=False)
"""
P.tight_layout()
P.show()
