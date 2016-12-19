#!/usr/bin/python
"""
Example script to realise a mock galaxy catalogue.
"""
import numpy as np
import pylab as P
import scipy.integrate
import sys
sys.path.append('src/')
import ghost

MHALO_MIN = 1e13 # Min. halo mass
NHALO = 100 # Number of halos to generate

np.random.seed(10)

# Load halo mass function (calculated using HMFcalc)
dat = np.genfromtxt("hmfcalc_massfn.dat").T
mhbin = dat[0]
dndlogm = dat[6]

# Keep only mass bins above the mass threshold
idxs = np.where(mhbin >= MHALO_MIN)
mhbin = mhbin[idxs]
dndlogm = dndlogm[idxs]

# Calculate normalisation
norm = scipy.integrate.simps(dndlogm, np.log(mhbin))

# Generate the correct number of halos
mhalo = []
for i in range(mhbin.size - 1):
    # Interpolate number density in this mass bin
    dm = mhbin[i+1] - mhbin[i]
    n = 0.5 * (dndlogm[i] + dndlogm[i+1]) * dm / (0.5 * (mhbin[i] + mhbin[i+1]))
    
    # Calculate no. of halos in this bin
    N = int( n * NHALO / norm )
    
    # Realise halo masses in this bin (with log-uniform distribution of masses)
    logmh = np.random.uniform( low=np.log(mhbin[i]), 
                               high=np.log(mhbin[i+1]), 
                               size=N )
    mhalo += [np.exp(logmh)]
mhalo = np.concatenate(mhalo)

params = {} #{'ms_cen_beta': 0.5}


z = np.zeros(mhalo.size)
mstar, sfr, passive = ghost.add_physical_properties(mhalo, z, params)

mag_int, mag_obs = ghost.add_optical_mags(mstar, sfr, z, 'u', params, atten=True)

P.plot(mag_int, mag_obs, 'r,')
P.xlim((-22., -4.))
P.ylim((-22., -4.))
P.show()

exit()
#help(ghost.add_optical_mags)

print np.where(passive == True)[0].size, np.where(passive == False)[0].size

P.plot(mstar, sfr, 'r,')
P.xscale('log')
P.yscale('log')

P.show()

#P.hist(np.log10(sfr), bins=50)
#P.yscale('log')
#P.show()

"""
# Compare mass function of generated halos with input mass function
P.hist(np.log(mhalos), bins=200, normed=True)
P.plot(np.log(mhbin), dndlogm / norm, 'r-', lw=2.)
P.yscale('log')
P.show()
"""
