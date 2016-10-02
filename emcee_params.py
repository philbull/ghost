#!/usr/bin/python
"""
Metropolis-Hastings MCMC to find best-fit parameters for the galaxy model.

$ mpirun --bind-to none -n 1 python emcee_params.py
"""
import numpy as np
#import pylab as P
import copy, sys, time
import emcee
#from emcee.utils import MPIPool
import galaxy_model as g
import likelihoods as like

np.random.seed(12)

NTHREADS = 64
NSAMPLES = 1000
BAND = 'z'
CHAIN_FILE = "chainx_opt_rad.dat"
nwalkers = 64 # nwalkers needs to be at least twice ndim (11*2 here)

# Set initial model parameters
#params0 = copy.copy(g.default_params)

# Hand-calibrated parameters (good for radio/opt, worse for SMF)
params0 = {'sfr_sfms_sigma': 0.4, 'mass_mstar_min': 6.0, 'opt_bands': ['u', 'g', 'r', 'i', 'z'], 'fpass_alpha0': 10.8, 'fpass_alpha1': 0.5, 'sfr_pass_pthres': 0.05, 'fpass_zeta': -0.3, 'ms_cen_nu': -0.72, 'mass_mstar_max': 14.0, 'opt_pdf_sigma': [0.28122634653850337, 0.25232918833346546, 0.2468073941298409, 0.25273681573440887, 0.2724513351999828], 'sfr_pass_beta': 0.37, 'mass_mhalo_max': 16.0, 'ms_cen_sigma1': 0.05, 'opt_offset': [-27.3176081, -25.9445857, -25.2836569, -24.981922, -24.8096689], 'opt_cross_amp': [-3.61194266, -2.70027825, -2.01635745, -1.72575487, -1.56393268], 'opt_mstar_beta': -0.000242854097, 'mhi_omegaHI': 0.0001, 'ms_cen_mu': 0.019, 'nsamp_mstar': 200, 'ms_cen_norm': 0.02, 'opt_cross_beta': -0.262636097, 'ms_cen_gamma0': 0.65, 'ms_cen_gamma1': -0.26, 'mass_mhalo_min': 5.0, 'ms_cen_xi': 4.25, 'opt_cross_gamma': 0.290177366, 'opt_mstar_amp': 3876.55879, 'ms_cen_logM2': 11.8, 'ms_cen_logM1': 12.1, 'sfr_pass_gamma': 1.9, 'sfr_min': -5.0, 'sfr_pass_mshift': 0.03, 'ms_cen_sigmainf': 0.16, 'sfr_pass_sfrmin': 1e-07, 'sfr_sfms_gamma': 1.9, 'sfr_pass_mscale': 1.0, 'sfr_sfms_mscale': 1.0, 'sfr_sfms_beta': 0.9, 'nsamp_mhalo': 200, 'sfr_sfms_alpha1': 1.07, 'sfr_sfms_alpha0': 0.005, 'opt_pdf_mean': [-0.06587481168919591, -0.053777765775930214, -0.01854712885192855, -0.008538656095465969, -0.008732300503716532], 'nsamp_sfr': 200, 'sfr_pass_sigma': 0.4, 'ms_cen_beta0': 1.4, 'ms_cen_beta1': 0.17, 'opt_mstar_c': -1.0108709, 'sfr_max': 4.0, 'fpass_beta': -1.1, 'sfr_pass_type': 'shifted sfms', 'sfr_pass_alpha1': 0.0, 'sfr_pass_alpha0': -5.0, 'mhi_vc1': 200.0, 'mhi_vc0': 50.0}

# Load 6dFGS local radio LF data from Mauch & Sadler (astro-ph/0612018)
L_radio, Phi_radio, errp_radio, errm_radio = \
                        like.load_mauch_lf(fname="lumfunc_6dfgs.dat", h=0.67)
logPhi_radio = np.log10(Phi_radio)

# Load GAMA local optical LF data from Driver et al. (2012)
mag_gama, Phi_gama, err_gama = like.load_gama_lf(band=BAND, h=0.67)

# Load SDSS-GALEX stellar mass function
mstar_sdss, sf_phi_sdss, sf_errp_sdss, sf_errm_sdss, \
            qu_phi_sdss, qu_errp_sdss, qu_errm_sdss = like.load_sdss_smf(h=0.67)
sf_logphi_sdss = np.log10(sf_phi_sdss)
qu_logphi_sdss = np.log10(qu_phi_sdss)

# Pre-calculate my halo model
pkfile = "camb_pk_z0.dat"
hm = g.HaloModel(pkfile, h=0.67, om=0.32)

#-------------------------------------------------------------------------------

def radio_lumfn(L, params, z=0.):
    """
    Calculate radio luminosity function for a given set of model parameters.
    """
    # Number density as a function of sfr, dn/dlog(sfr)
    sfr = L * 5.52e-29 # erg/s/Hz, Bell (2003), Eq. 6
    dndlogsfr_sfms, dndlogsfr_pass = g.sfr_fn(hm, sfr, z=z, params=params)
    phi = dndlogsfr_sfms #+ dndlogsfr_pass # FIXME: Remove passive galaxies
    return phi

def optical_lumfn(obsmag, band, params, z=0.):
    """
    Calculate optical luminosity function in some band, for a given set of 
    model parameters.
    """
    #intmag = np.linspace()
    #dndmag_sfms_dust, dndmag_pass_dust = g.optical_mag_fn_dust( 
    #                                        hm, obsmag, intmag, band=band, 
    #                                        z=z, params=params )
    #phi = dndmag_sfms_dust + dndmag_pass_dust
    
    dndmag_sfms, dndmag_pass = g.optical_mag_fn( hm, obsmag, band, z=z, 
                                                 params=params )
    phi = dndmag_sfms + dndmag_pass
    return phi

def loglike_barlow(log10model, log10y, errp, errm):
    """
    Use the likelihood approximation for asymmetric error bars, from Barlow 
    [arXiv:physics/0306138]. Assumes that logy and logmodel are log_10, and 
    the errorbars are the log-space errorbars.
    """
    sigp = np.abs(errp)
    sigm = np.abs(errm)
    sig = 0.5 * (sigp + sigm)
    A = (sigp - sigm) / (sigp + sigm)
    x = (log10y - log10model) / sig # FIXME: minus sign?
    chi2 = x**2. * ( 1. - 2.*A*x + 5.*(A*x)**2. )
    return -0.5 * np.sum(chi2)

def loglike_radio_lf(L, logPhi, errp, errm, params):
    """
    Log-likelihood for the radio luminosity function.
    Use the likelihood approximation for asymmetric error bars, from Barlow 
    [arXiv:physics/0306138].
    """
    mphi = radio_lumfn(L, params)
    return loglike_barlow(np.log10(mphi), logPhi, errp, errm)

def loglike_smf(mstar, sf_logPhi, sf_errp, sf_errm, 
                pass_logPhi, pass_errp, pass_errm, params, z=0.):
    """
    Log-likelihood for the stellar mass function. Requires both star-forming 
    and passive MFs.
    """
    # Calculate stellar mass function and passive fraction
    dndlogms = g.stellar_mass_fn(hm, mstar, z=z, params=params)
    fpass = g.f_passive(mstar, z=z, params=params)
    
    # Calculate likelihoods using asymmetric errorbars
    logl_sf = loglike_barlow( np.log10((1.-fpass)*dndlogms), 
                              sf_logPhi, sf_errp, sf_errm )
    
    logl_pass = loglike_barlow( np.log10(fpass*dndlogms), 
                                pass_logPhi, pass_errp, pass_errm )
    return logl_sf + logl_pass

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
    
    # Apply priors to tricky parameters
    if p['sfr_sfms_sigma'] > 1.0:  return -np.inf
    if p['sfr_sfms_sigma'] < 0.05: return -np.inf
    if p['sfr_pass_sigma'] > 2.:   return -np.inf
    if p['sfr_pass_sigma'] < 0.01: return -np.inf
    #if p['sfr_sfms_alpha0'] > 1.:  return -np.inf
    #if p['sfr_pass_alpha0'] > 2.: return -np.inf
    #if p['sfr_pass_alpha0'] > 2.: return -np.inf
    if p['sfr_pass_mshift'] > 0.9: return -np.inf
    if p['sfr_pass_mshift'] < 1e-3: return -np.inf
    
    if p['ms_cen_logM1'] < 11.6: return -np.inf
    if p['ms_cen_logM1'] > 12.5: return -np.inf
    # Make sure passive and active sequences are mostly disjoint
    #if p['sfr_pass_mshift'] > 0.9 * 10.**(-p['sfr_pass_sigma']): return -np.inf
    
    # Calculate log-likelihoods for each dataset
    try:
        # Radio luminosity function
        logl_rad = loglike_radio_lf(L_radio, logPhi_radio, errp_radio, 
                                    errm_radio, params=p)
        # Optical luminosity function
        logl_opt = loglike_optical_lf(mag_gama, Phi_gama, err_gama, band=BAND, 
                                      params=p)
        # Stellar mass function (star-forming and passive)
        logl_smf = loglike_smf( mstar_sdss, 
                                sf_logphi_sdss, sf_errp_sdss, sf_errm_sdss,
                                qu_logphi_sdss, qu_errp_sdss, qu_errm_sdss,
                                params=p, z=0.)
        print logl_rad, logl_opt, logl_smf
        #print logl_rad, logl_smf
        #print logl_smf
    except BaseException as e:
        print e
        return -np.inf
    
    # Sanity checks
    logl = logl_rad + logl_opt #+ logl_smf/5. # FIXME: Downweighted SMF
    #logl = logl_rad + logl_smf / 10.
    #logl = logl_smf
    if np.isnan(logl): logl = -np.inf
    #print logl
    return logl

#-------------------------------------------------------------------------------

# Specify parameters to sample and initial values
#pnames = ['sfr_sfms_alpha0', 'sfr_sfms_beta', 'sfr_sfms_sigma', 
#          'sfr_pass_alpha0', 'sfr_pass_beta', 'sfr_pass_sigma', 
#          'ms_cen_norm', 'ms_cen_logM1', 'ms_cen_beta0', 'ms_cen_gamma0',
#          'ms_cen_sigmainf', 'ms_cen_sigma1',
#          'fpass_alpha0', 'fpass_beta', 'fpass_zeta',
#          ]
#pnames = ['ms_cen_norm', 'ms_cen_logM1', 'ms_cen_beta0', 'ms_cen_gamma0',
#          'ms_cen_sigmainf', 'ms_cen_sigma1',
#          'fpass_alpha0', 'fpass_beta', 'fpass_zeta',
#          ]
#pnames = ['sfr_sfms_alpha0', 'sfr_sfms_beta', 'sfr_sfms_sigma', 
#          'sfr_pass_alpha0', 'sfr_pass_beta', 'sfr_pass_sigma', 
#          'fpass_alpha0', 'fpass_beta', 'fpass_zeta',
#          ]
#pnames = ['sfr_sfms_alpha0', 'sfr_sfms_beta', 'sfr_sfms_sigma', 
#          'sfr_pass_mshift', 'sfr_pass_sigma', 
#          'ms_cen_norm', 'ms_cen_logM1', 'ms_cen_beta0', 'ms_cen_gamma0',
#          'ms_cen_sigmainf', 'ms_cen_sigma1',
#          'fpass_alpha0', 'fpass_beta', 'fpass_zeta',
#          ]
# SMF-params only
#pnames = ['fpass_alpha0', 'fpass_beta', 'fpass_zeta',
#          'ms_cen_beta0', 'ms_cen_gamma0', 'ms_cen_logM1',
#          'ms_cen_norm', 'ms_cen_sigma1', 'ms_cen_sigmainf',
#          'ms_cen_logM2', 'ms_cen_xi' ]

# Most SMF params, all SFR params
pnames = ['sfr_sfms_alpha0', 'sfr_sfms_beta', 'sfr_sfms_sigma', 
          'sfr_pass_mshift', 'sfr_pass_sigma', 
          'ms_cen_norm', 'ms_cen_logM1', 'ms_cen_beta0', 'ms_cen_gamma0',
          'ms_cen_sigmainf', 'ms_cen_sigma1',
          'fpass_alpha0', 'fpass_beta', 'fpass_zeta',
          ]

p0 = np.array([params0[pp] for pp in pnames])
ndim = p0.size

# Get random initial positions for walkers (best-fit values x some O(1) factor)
p0 = np.outer(np.ones(nwalkers), p0)
p0 *= np.random.normal(loc=1., scale=0.0005, size=p0.shape)

# Set up MPI pooling
#pool = MPIPool()
#if not pool.is_master():
#    pool.wait()
#    sys.exit(0)

# Initialise emcee sampler and write header of chain file
sampler = emcee.EnsembleSampler(nwalkers, ndim, loglike, 
                           args=(pnames, params0), threads=NTHREADS) #pool=pool)
f = open(CHAIN_FILE, "w")
f.write("# %s %s %s\n" % ("walker", "logl", " ".join(pnames)))
f.close()

# Iterate over samples
nsteps = NSAMPLES
tstart = time.time()
print "Starting %d samples with %d walkers and %d threads." \
       % (nsteps, nwalkers, NTHREADS)
for i, result in enumerate(sampler.sample(p0, iterations=nsteps)):
    
    # Save current sample
    position = result[0]
    prob = result[1]
    f = open(CHAIN_FILE, "a")
    for k in range(nwalkers):
        pvals = " ".join(["%s" % x for x in position[k]])
        f.write("%d %f %s\n" % (k, prob[k], pvals))
    f.close()
    
    # Print status
    print "Step %d / %d done in %3.1f sec" % (i+1, nsteps, time.time() - tstart)
    tstart = time.time()
#pos, prob, state = sampler.run_mcmc(p0, 10)

# Print diagnostics
#print sampler.acceptance_fraction()

# Clean up MPI processes
#pool.close()
sys.exit(0)
