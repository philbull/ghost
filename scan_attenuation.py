#!/usr/bin/python
"""
Find best-fit parameters for the optical attenuation amplitude by scanning 
through values of the amplitude.
"""
import numpy as np
import pylab as P
import scipy.interpolate
import scipy.integrate
import scipy.optimize
import copy, sys, time
import galaxy_model as g
import likelihoods as like
from bestfit import *

#np.random.seed(15)

# Get band from commandline
try:
    BAND = str(sys.argv[1])
    print "Fitting band", BAND
except:
    print "Error: Need to specify which band to fit for."
    sys.exit(1)

MSTAR_MIN = None
HUBBLE = 0.67
OMEGA_M = 0.32

# Set initial model parameters
#params0 = copy.copy(g.default_params)
#params0 = {'sfr_sfms_sigma': 0.415595676119, 'mass_mstar_min': 6.0, 'opt_bands': ['u', 'g', 'r', 'i', 'z'], 'fpass_alpha0': 10.839733046599999, 'fpass_alpha1': 0.5, 'sfr_pass_pthres': 0.05, 'fpass_zeta': -0.95027169466200001, 'ms_cen_nu': -0.72, 'mass_mstar_max': 14.0, 'opt_pdf_sigma': [0.28122634653850337, 0.25232918833346546, 0.2468073941298409, 0.25273681573440887, 0.2724513351999828], 'extinction_alpha': -7.57886, 'extinction_beta': 5.3450337472999996, 'sfr_pass_beta': 0.37, 'extinction_diskfac': 10.2412022264, 'mass_mhalo_max': 16.0, 'ms_cen_sigma1': 0.56640755541800003, 'opt_offset': [-27.3176081, -25.9445857, -25.2836569, -24.981922, -24.8096689], 'opt_cross_amp': [-3.61194266, -2.70027825, -2.01635745, -1.72575487, -1.56393268], 'opt_mstar_beta': -0.000242854097, 'mhi_omegaHI': 0.0001, 'ms_cen_mu': 0.019, 'nsamp_mstar': 200, 'ms_cen_norm': 0.015185988130400001, 'opt_cross_beta': -0.262636097, 'ms_cen_gamma0': 0.56119468698499997, 'ms_cen_gamma1': -0.26, 'extinction_amp': 0.33240776116999998, 'mass_mhalo_min': 7.0, 'ms_cen_xi': 4.25, 'opt_cross_gamma': 0.290177366, 'opt_mstar_amp': 3876.55879, 'ms_cen_logM2': 11.8, 'ms_cen_logM1': 12.488922480999999, 'sfr_pass_gamma': 1.9, 'sfr_min': -9.0, 'sfr_pass_mshift': 0.0083967790705399992, 'ms_cen_sigmainf': 0.025222009908500001, 'sfr_pass_sfrmin': 1e-07, 'sfr_sfms_gamma': 1.9, 'sfr_pass_mscale': 1.0, 'sfr_sfms_mscale': 1.0, 'sfr_sfms_beta': 0.99195297218300005, 'nsamp_mhalo': 200, 'sfr_sfms_alpha1': 1.07, 'sfr_sfms_alpha0': -0.0581345697711, 'opt_pdf_mean': [-0.06587481168919591, -0.053777765775930214, -0.01854712885192855, -0.008538656095465969, -0.008732300503716532], 'nsamp_sfr': 200, 'sfr_pass_sigma': 0.011843956765499999, 'ms_cen_beta0': 1.2756497875999999, 'ms_cen_beta1': 0.17, 'opt_mstar_c': -1.0108709, 'sfr_max': 4.0, 'fpass_beta': -2.7139265055099999, 'sfr_pass_type': 'shifted sfms', 'sfr_pass_alpha1': 0.0, 'sfr_pass_alpha0': -5.0, 'mhi_vc1': 200.0, 'mhi_vc0': 50.0}

params0 = params_bf

# Set frequency scaling to be flat, so that resulting amplitude is 
# band-independent (but set to be close to the current amplitude at that band)
scaled_tau0 = params0['extinction_tau0'] \
            * np.exp(-params0['extinction_kappa'] \
                     * (g.band_wavelength[BAND] - params0['extinction_lambda0']))
#params0['extinction_tau0'] = scaled_tau0
params0['extinction_kappa'] = 0.


# Load GAMA local optical LF data from Driver et al. (2012)
data_gama = []
mag_gama, Phi_gama, err_gama = like.load_gama_lf(band=BAND, h=HUBBLE)
# FIXME: Apply magnitude cut to avoid incompleteness effects
mag_gama = mag_gama[:-3]
Phi_gama = Phi_gama[:-3]
err_gama = err_gama[:-3]
#data_gama.append([mag_gama, Phi_gama, err_gama])

# Pre-calculate my halo model
pkfile = "camb_pk_z0.dat"
hm = g.HaloModel(pkfile, h=HUBBLE, om=OMEGA_M)

#-------------------------------------------------------------------------------

def optical_lumfn(obsmag, band, params, z=0.):
    """
    Calculate optical luminosity function in some band, for a given set of 
    model parameters.
    """
    # Include dust attenuation correction in the optical
    dndmag_sfms, dndmag_pass = g.optical_mag_fn_atten( hm, obsmag, band, 
                                                       z=z, params=params )
    phi = dndmag_sfms + dndmag_pass
    return phi

def loglike_optical_lf(mag, phi, err, band, params, z=0.):
    """
    Calculate log-likelihood for optical luminosity function.
    """
    mphi = optical_lumfn(mag, band, params, z=z)
    logl = -0.5 * np.sum( (phi - mphi)**2. / err**2. )
    return logl

def loglike(pvals, pnames, params0):
    """
    Evaluate total log-likelihood for the set of input parameter values.
    """
    # Build parameter dictionary
    p = copy.copy(params0)
    for i in range(len(pnames)): p[pnames[i]] = pvals[i]
    
    # Calculate log-likelihoods for each dataset
    try:
        # Optical luminosity function, per-band
        #mag_gama, Phi_gama, err_gama = data
        logl_opt = loglike_optical_lf(mag_gama, Phi_gama, err_gama, 
                                      band=BAND, params=p)
        print logl_opt
    except BaseException as e:
        print e
        return -np.inf
    
    # Sanity checks
    logl = np.sum(logl_opt)
    if np.isnan(logl): logl = -np.inf
    #print logl
    return logl

#-------------------------------------------------------------------------------

print "Scaled amplitude for %s band: %3.3e" % (BAND, scaled_tau0)

#pnames = ['extinction_amp',]
pnames = ['extinction_tau0',]
pvals = scaled_tau0 #* np.logspace(-1., 1., 30)

if BAND == 'u':
    pvals *= np.logspace(-1., 1., 30)
elif BAND == 'g':
    pvals *= np.logspace(-1., 1., 30)
elif BAND == 'r':
    pvals *= np.logspace(-1., 2., 30)
elif BAND == 'i':
    pvals *= np.logspace(-3., 0.5, 30)
elif BAND == 'z':
    pvals *= np.logspace(-2., 2., 30)


# Scan through parameter values
logl = [loglike([v,], pnames, params0) for v in pvals]
logl_interp = scipy.interpolate.interp1d(np.log(pvals), logl, 
                                         kind='quadratic', bounds_error=False)
# Upper bound can miss (rounding error?) sometimes; correct by 0.9999 factor
pp = np.logspace(np.log10(np.min(pvals)), np.log10(np.max(pvals)*0.9999), 1000)
_logl = logl_interp(np.log(pp))

# Calculate cumulative dist. fn.
cdf = scipy.integrate.cumtrapz(np.exp(_logl - np.max(_logl)), pp, initial=0.)
cdf /= cdf[-1]
p_cdf = scipy.interpolate.interp1d(cdf, pp, kind='linear')

## Fit quadratic function
#def lsq(params):
#    a, b, c = params
#    return _logl - (a * (np.log(pp) - b)**2. + c)
#params0 = [-1., np.log(np.mean(pvals)), 0.]
#pnew = scipy.optimize.leastsq(lsq, params0)[0]
#print pnew

# Save likelihoods for later
np.save("atten_like_%s" % BAND, np.column_stack((pvals, logl)))

# Calculate median and 1/2-sigma bounds
print "%3.3e %3.3e %3.3e %3.3e %3.3e" % (p_cdf(0.5), p_cdf(0.16), p_cdf(0.84), 
                             p_cdf(0.5)- p_cdf(0.16), p_cdf(0.84)- p_cdf(0.5))

print "%s-band best-fit amp: %4.4e (logL = %4.1f)" \
    % (BAND, pp[np.where(_logl == np.max(_logl))][0], np.max(_logl))

exit()
###########

P.subplot(111)
P.title(BAND)
P.plot(pvals, logl, 'b', lw=1.8, marker='.')

#P.plot(pp, pnew[0]*(np.log(pp) - pnew[1])**2. + pnew[2], 'g-', lw=1.8)

P.plot(pp, cdf, 'y-', lw=1.8)
P.plot(pp, _logl, 'r-', lw=1.8)
P.xscale('log')
P.tight_layout()
P.show()
