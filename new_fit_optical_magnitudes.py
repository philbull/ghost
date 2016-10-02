#!/usr/bin/python
"""
Get best-fit parameters for fitting relation of Guo optical magnitudes as 
a function of SFR and Mstar. Perform a global fit to all bands.
"""
import numpy as np
import pylab as P
from scipy.optimize import leastsq
import scipy.integrate

NBINS = 60
#DATAFILE = '../delucia_mass/guo_colours_z0.dat'
#DATAFILE = '../delucia_mass/guo_colours_z0.24.dat'
#DATAFILE = '../delucia_mass/guo_colours_z0.51.dat'
#DATAFILE = '../delucia_mass/guo_colours_z0.76.dat'
#DATAFILE = '../delucia_mass/guo_colours_z0.99.dat'
#DATAFILE = '../delucia_mass/guo_colours_z1.50.dat'
#DATAFILE = '../delucia_mass/guo_colours_z2.07.dat'
DATAFILE = '../delucia_mass/guo_colours_z3.06.dat'

# Guo (2010)
mh, mstar, sfr, mag_u, mag_g, mag_r, mag_i, mag_z = \
    np.genfromtxt(DATAFILE, delimiter=',', skip_header=1).T
mags = [mag_u, mag_g, mag_r, mag_i, mag_z]
names = ['u', 'g', 'r', 'i', 'z']
colours = ['b', 'g', 'r', 'y', 'm']

# Convert units
mh *= 1e10 / 0.73
mstar *= 1e10 / 0.73

# Apply cuts for fitting
idxs = np.where(np.logical_and.reduce((
                    sfr > 1e-7,
                    mstar > 0.,
                    mags[0] < 0.,
                    mags[1] < 0.,
                    mags[2] < 0.,
                    mags[3] < 0.,
                    mags[4] < 0.)) )

def mag_fit(p, cat=True):
    """
    Magnitude fitting function.
    """
    tot = []
    for i in range(len(mags)):
        #y = mags[i][idxs] \
        #  - p[0] * (p[1] + (mstar[idxs]/1e9)**p[2]) \
        #  - p[3 + i*3]*(mstar[idxs]/1e9)**p[4 + i*3] \
        #  * sfr[idxs]**p[5 + i*3]
        y = mags[i][idxs] \
          - p[0] * (p[1] + (mstar[idxs]/1e9)**p[2]) \
          - p[5 + i*2]*(mstar[idxs]/1e9)**p[3] \
            * sfr[idxs]**p[4] \
          + p[6 + i*2]
        tot.append(y)
    if cat:
        return np.concatenate(tot)
    else:
        return tot

def logn(x, mu, sigma, c):
    """
    Log-normal function.
    """
    u = c - x
    return np.exp(-0.5 * ((np.log(u) - mu) / sigma)**2.) \
          / (np.sqrt(2.*np.pi) * u * sigma)

def lognormal(p, x, y):
    """
    Log-normal distribution pdf.
    """
    mu, sigma = p
    c = 1.
    u = c - x
    pdf = np.exp(-0.5 * ((np.log(u) - mu) / sigma)**2.) \
          / (np.sqrt(2.*np.pi) * u * sigma)
    return y - pdf

p0 = [  
   5.33523486e+02,  -1.02793533e+00,  -1.67491369e-03,  -0.1, -0.1,
  -3.3, 0.,
  -3.3, 0.,
  -3.3, 0.,
  -3.3, 0.,
  -3.3, 0., ]

# New new
#sigma = 0.274
p0 = [  3.87655879e+03,  -1.01087090e+00,  -2.42854097e-04,  -2.62636097e-01,
   2.90177366e-01,  -3.61194266e+00,  -2.73176081e+01,  -2.70027825e+00,
  -2.59445857e+01,  -2.01635745e+00,  -2.52836569e+01,  -1.72575487e+00,
  -2.49819220e+01,  -1.56393268e+00,  -2.48096689e+01,]

# This bit does the fitting
p = leastsq(mag_fit, p0)[0]
#p = p0

mag_est = mag_fit(p)
sigma = np.std(mag_est)
print "sigma = %3.3f" % sigma
print p
resid = mag_fit(p, cat=False)


#-------------------------------------------------------------------------------
# Fit log-normal distributions
#-------------------------------------------------------------------------------

# Loop over bands and fit log-normal to residual histogram
P.subplot(111)
ln_mean = []; ln_sigma = []
for i in range(len(mags)):
    # Get histogram
    pdf, edges = np.histogram(resid[i], bins=NBINS, range=(-1.5, 1.), normed=True)
    xc = 0.5 * (edges[1:] + edges[:-1])
    
    # Fit log-normal to residuals
    p0 = [0., 0.25,]
    pln = leastsq(lognormal, p0, args=(xc, pdf))[0]
    
    # Plot 
    x = np.linspace(-2.5, 0.9999, 500)
    P.plot(x, logn(x, pln[0], pln[1], 1.), color=colours[i], lw=2., ls='dashed')
    print i, pln
    ln_mean.append(pln[0])
    ln_sigma.append(pln[1])

    print "...sigma(%d) = %3.3f" % (i, np.std(resid[i]))
    P.hist(resid[i], bins=NBINS, range=(-1.5, 1.), normed=True, histtype='step', 
           lw=2.8, label=names[i], color=colours[i], alpha=0.5)
    
    # Check normalisation
    I = scipy.integrate.simps(logn(x, pln[0], pln[1], 1.), x)
    print "...Norm = %5.5f" % I

#-------------------------------------------------------------------------------
# Map parameter values to dictionary
#-------------------------------------------------------------------------------

# Construct parameter name list
params = {
    'opt_mstar_amp':    p[0],
    'opt_mstar_c':      p[1],
    'opt_mstar_beta':   p[2],
    'opt_cross_beta':   p[3],
    'opt_cross_gamma':  p[4],
    'opt_cross_amp':    [p[5 + i*2] for i in range(len(names))],
    'opt_offset':       [p[6 + i*2] for i in range(len(names))],
    'opt_pdf_mean':     ln_mean,
    'opt_pdf_sigma':    ln_sigma,
    'opt_bands':        names
}
print DATAFILE
print "-"*50
knames = params.keys()
knames.sort()
for key in knames:
    print "'%s': %s," % (key, params[key])



P.legend(loc='upper right')
P.tight_layout()
P.show()
