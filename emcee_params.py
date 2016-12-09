#!/usr/bin/python
"""
Metropolis-Hastings MCMC to find best-fit parameters for the galaxy model.

$ mpirun --bind-to none -n 1 python emcee_params.py
"""
import numpy as np
import copy, sys, time
import emcee
#from emcee.utils import MPIPool
import galaxy_model as g
import likelihoods as like

np.random.seed(18) #17 #12

NTHREADS = 64
NSAMPLES = 2000 #200 #1000
BANDS = ['u', 'g', 'r', 'i', 'z']
#BANDS = ['g', 'z']
#CHAIN_FILE = "chainx_opt_rad.dat"
#CHAIN_FILE = "chainx_opt_rad_atten.dat"
#CHAIN_FILE = "chain_new_gzrad_atten.dat"
#CHAIN_FILE = "chain_new_gzrad_smf_atten.dat"
CHAIN_FILE = "chain_gzrad_atten_3.dat"
nwalkers = 128 #64 # nwalkers needs to be at least twice ndim (11*2 here)

MSTAR_MIN = None
HUBBLE = 0.67
OMEGA_M = 0.32

# Set initial model parameters
#params0 = copy.copy(g.default_params)
#params0 = {'sfr_sfms_sigma': 0.38750005196600001, 'mass_mstar_min': 6.0, 'opt_bands': ['u', 'g', 'r', 'i', 'z'], 'fpass_alpha0': 10.8205970498, 'fpass_alpha1': 0.5, 'sfr_pass_pthres': 0.05, 'fpass_zeta': -1.5353087936700001, 'ms_cen_nu': -0.72, 'mass_mstar_max': 14.0, 'opt_pdf_sigma': [0.28122634653850337, 0.25232918833346546, 0.2468073941298409, 0.25273681573440887, 0.2724513351999828], 'extinction_alpha': -7.57886, 'extinction_beta': 5.4223162433100001, 'sfr_pass_beta': 0.37, 'extinction_diskfac': 13.516079320399999, 'mass_mhalo_max': 16.0, 'ms_cen_sigma1': 0.60802361084000001, 'opt_offset': [-27.3176081, -25.9445857, -25.2836569, -24.981922, -24.8096689], 'ms_cen_logM1': 12.489042828500001, 'mhi_omegaHI': 0.0001, 'ms_cen_mu': 0.019, 'nsamp_mstar': 300, 'ms_cen_norm': 0.015799159819799999, 'nsamp_mhalo': 200, 'opt_cross_beta': -0.262636097, 'ms_cen_gamma0': 0.54792407909100005, 'ms_cen_gamma1': -0.26, 'extinction_amp': 0.29403960120099998, 'mass_mhalo_min': 7.0, 'ms_cen_xi': 4.25, 'opt_cross_gamma': 0.290177366, 'opt_mstar_amp': 3876.55879, 'ms_cen_logM2': 11.8, 'opt_mstar_beta': -0.000242854097, 'sfr_pass_type': 'shifted sfms', 'sfr_min': -9.0, 'sfr_pass_mshift': 0.0053486148026400003, 'ms_cen_sigmainf': -0.031605451195699998, 'sfr_pass_alpha1': 0.0, 'sfr_sfms_gamma': 1.9, 'nsamp_sfr': 250, 'sfr_sfms_mscale': 1.0, 'sfr_sfms_beta': 1.0910253563500001, 'opt_cross_amp': [-3.61194266, -2.70027825, -2.01635745, -1.72575487, -1.56393268], 'sfr_sfms_alpha1': 1.07, 'sfr_sfms_alpha0': -0.10162165209600001, 'opt_pdf_mean': [-0.06587481168919591, -0.053777765775930214, -0.01854712885192855, -0.008538656095465969, -0.008732300503716532], 'sfr_pass_mscale': 1.0, 'sfr_pass_sigma': 0.012386348129800001, 'ms_cen_beta0': 1.29481129595, 'ms_cen_beta1': 0.17, 'opt_mstar_c': -1.0108709, 'sfr_max': 4.0, 'fpass_beta': -2.7709240453000001, 'sfr_pass_gamma': 1.9, 'sfr_pass_sfrmin': 1e-07, 'sfr_pass_alpha0': -5.0, 'mhi_vc1': 200.0, 'mhi_vc0': 50.0, 'extinction_tau0': 0.452}
from bestfit import *
params0 = params_bf

# Load 6dFGS local radio LF data from Mauch & Sadler (astro-ph/0612018)
L_radio, Phi_radio, errp_radio, errm_radio = \
               like.load_mauch_lf(fname="../lumfns/lumfunc_6dfgs.dat", h=HUBBLE)
logPhi_radio = np.log10(Phi_radio)

# Load GAMA local optical LF data from Driver et al. (2012)
data_gama = []
for BAND in BANDS:
    mag_gama, Phi_gama, err_gama = like.load_gama_lf(band=BAND, h=HUBBLE)
    # FIXME: Apply magnitude cut to avoid incompleteness effects
    mag_gama = mag_gama[:-3]
    Phi_gama = Phi_gama[:-3]
    err_gama = err_gama[:-3]
    data_gama.append([mag_gama, Phi_gama, err_gama])

# Load SDSS-GALEX stellar mass function
mstar_sdss, sf_phi_sdss, sf_errp_sdss, sf_errm_sdss, \
            qu_phi_sdss, qu_errp_sdss, qu_errm_sdss \
                             = like.load_sdss_smf(h=HUBBLE, mstar_min=MSTAR_MIN)
# Trim low-mass datapoints
mstar_sdss, sf_phi_sdss, sf_errp_sdss, sf_errm_sdss, \
qu_phi_sdss, qu_errp_sdss, qu_errm_sdss \
  = [var[4:] for var in (mstar_sdss, sf_phi_sdss, sf_errp_sdss, sf_errm_sdss,
                         qu_phi_sdss, qu_errp_sdss, qu_errm_sdss)]
sf_logphi_sdss = np.log10(sf_phi_sdss)
qu_logphi_sdss = np.log10(qu_phi_sdss)

# Pre-calculate my halo model
pkfile = "camb_pk_z0.dat"
hm = g.HaloModel(pkfile, h=HUBBLE, om=OMEGA_M)

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

def optical_lumfn(obsmag, band, params, z=0., atten=True):
    """
    Calculate optical luminosity function in some band, for a given set of 
    model parameters.
    """
    # Decide whether dust-attenuated optical luminosity fn. should be used
    if atten and band != 'z': # FIXME: Temporary hack to ignore atten. for z-band
        # Include dust attenuation correction in the optical
        dndmag_sfms, dndmag_pass = g.optical_mag_fn_atten( hm, obsmag, band, 
                                                           z=z, params=params )
    else:
        # No dust attenuation in the optical
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
    if p['sfr_pass_mshift'] > 0.9: return -np.inf
    if p['sfr_pass_mshift'] < 1e-3: return -np.inf
    if p['ms_cen_sigma1'] < 2e-2: return -np.inf
    if p['ms_cen_sigmainf'] < 2e-2: return -np.inf
    if p['ms_cen_logM1'] < 11.6: return -np.inf
    if p['ms_cen_logM1'] > 12.5: return -np.inf
    
    # Calculate log-likelihoods for each dataset
    try:
        # Radio luminosity function
        logl_rad = loglike_radio_lf(L_radio, logPhi_radio, errp_radio, 
                                    errm_radio, params=p)
        
        # Optical luminosity function, per-band
        logl_opt = []
        for BAND, data in zip(BANDS, data_gama):
            mag_gama, Phi_gama, err_gama = data
            _logl_opt = loglike_optical_lf(mag_gama, Phi_gama, err_gama, 
                                           band=BAND, params=p)
            logl_opt.append(_logl_opt)
            
        # Stellar mass function (star-forming and passive)
        #logl_smf = loglike_smf( mstar_sdss, 
        #                        sf_logphi_sdss, sf_errp_sdss, sf_errm_sdss,
        #                        qu_logphi_sdss, qu_errp_sdss, qu_errm_sdss,
        #                        params=p, z=0.)
        
        # Output likelihood report
        print logl_opt, logl_rad, "||", np.sum(logl_opt) + logl_rad
    except BaseException as e:
        print "Error:", e
        return -np.inf
    
    # Sanity checks
    logl = np.sum(logl_opt) + logl_rad #+ logl_smf
    if np.isnan(logl): logl = -np.inf
    return logl


#-------------------------------------------------------------------------------

# Specify parameters to sample and initial values
pnames = ['sfr_sfms_alpha0', 'sfr_sfms_beta', 'sfr_sfms_sigma', 
          'sfr_pass_mshift', 'sfr_pass_sigma', 
          'ms_cen_norm', 'ms_cen_logM1', 'ms_cen_beta0', 'ms_cen_gamma0',
          'ms_cen_sigmainf', 'ms_cen_sigma1',
          'fpass_alpha0', 'fpass_beta', 'fpass_zeta',
          'extinction_tau0', 'extinction_beta', 'extinction_diskfac', 
          ]
          
p0 = np.array([params0[pp] for pp in pnames])
ndim = p0.size

logl0 = loglike(p0, pnames, params0)
#exit()

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
