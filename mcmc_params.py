#!/usr/bin/python
"""
Metropolis-Hastings MCMC to find best-fit parameters for the galaxy model.
"""
import numpy as np
import pylab as P
import copy
import galaxy_model as g
import likelihoods as like

np.random.seed(12)

NSAMPLES = 100
BAND = 'z'

# Set initial model parameters
params = copy.copy(g.default_params)

params = {'opt_mstar_amp': 3876.55879, 'sfr_sfms_sigma': 0.45, 'mass_mstar_min': 6.0, 'ms_cen_logM2': 11.8045, 'opt_mstar_beta': -0.000242854097, 'opt_bands': ['u', 'g', 'r', 'i', 'z'], 'fpass_alpha0': 10.4, 'fpass_alpha1': 0.5, 'sfr_pass_pthres': 0.05, 'sfr_min': -5.0, 'ms_cen_nu': -0.72, 'mass_mstar_max': 14.0, 'sfr_pass_mshift': 0.6, 'opt_pdf_sigma': [0.28122634653850337, 0.25232918833346546, 0.2468073941298409, 0.25273681573440887, 0.2724513351999828], 'mass_mhalo_max': 16.0, 'ms_cen_gamma0': 0.45, 'sfr_pass_logu': False, 'sfr_sfms_beta': 0.7702552880588892, 'opt_cross_amp': [-3.61194266, -2.70027825, -2.01635745, -1.72575487, -1.56393268], 'sfr_sfms_alpha1': 1.07, 'sfr_sfms_alpha0': -7.898308056959661, 'opt_pdf_mean': [-0.06587481168919591, -0.053777765775930214, -0.01854712885192855, -0.008538656095465969, -0.008732300503716532], 'ms_cen_sigma1': 0.046, 'nsamp_sfr': 200, 'opt_offset': [-27.3176081, -25.9445857, -25.2836569, -24.981922, -24.8096689], 'sfr_pass_sigma': 0.5160599470876794, 'ms_cen_sigmainf': 0.1592, 'ms_cen_logM1': 12.008643166868417, 'mhi_omegaHI': 0.0001, 'ms_cen_beta0': 1.6212566216867343, 'ms_cen_beta1': 0.17, 'opt_mstar_c': -1.0108709, 'sfr_max': 4.0, 'ms_cen_mu': 0.019, 'nsamp_mstar': 200, 'ms_cen_norm': 0.013690232619302729, 'fpass_beta': -1.3, 'nsamp_mhalo': 200, 'opt_cross_beta': -0.262636097, 'sfr_pass_sfrmin': 1e-07, 'ms_cen_gamma1': -0.26, 'mass_mhalo_min': 5.0, 'ms_cen_xi': 4.2503, 'opt_cross_gamma': 0.290177366, 'mhi_vc1': 200.0, 'mhi_vc0': 50.0}




# Load 6dFGS local radio LF data from Mauch & Sadler (astro-ph/0612018)
L_radio, Phi_radio, errp_radio, errm_radio = \
                        like.load_mauch_lf(fname="lumfunc_6dfgs.dat", h=0.67)
logPhi_radio = np.log10(Phi_radio)

# Load GAMA local optical LF data from Driver et al. (2012)
mag_gama, Phi_gama, err_gama = like.load_gama_lf(band=BAND, h=0.67)

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
    phi = dndlogsfr_sfms + dndlogsfr_pass
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
    
def loglike(L, phi, err, params):
    """
    Calculate log-likelihood.
    """
    mphi = radio_lumfn(L, params)
    logl = -0.5 * np.sum( (phi - mphi)**2. / err**2. )
    
    #lbl = "beta = %2.2f, logL = %3.3e" % (params['sfr_sfms_beta'], logl)
    #P.plot(L, mphi, alpha=0.5, label=lbl)
    return logl

def loglike_barlow(L, logPhi, errp, errm, params):
    """
    Use the likelihood approximation for asymmetric error bars, from Barlow 
    [arXiv:physics/0306138].
    """
    mphi = radio_lumfn(L, params)
    
    sigp = np.abs(errp)
    sigm = np.abs(errm)
    sig = 0.5 * (sigp + sigm)
    A = (sigp - sigm) / (sigp + sigm)
    x = (logPhi - np.log10(mphi)) / sig # FIXME: minus sign?
    chi2 = x**2. * ( 1. - 2.*A*x + 5.*(A*x)**2. )
    return -0.5 * np.sum(chi2)

def loglike_optical(mag, phi, err, band, params, z=0.):
    """
    Calculate log-likelihood for optical luminosity function.
    """
    mphi = optical_lumfn(mag, band, params, z=z)
    logl = -0.5 * np.sum( (phi - mphi)**2. / err**2. )
    return logl

#-------------------------------------------------------------------------------

# Set up proposal distribution and initial set of params
pnames = ['sfr_sfms_beta', 'sfr_sfms_sigma', 'sfr_sfms_alpha0', 
          'ms_cen_norm', 'ms_cen_logM1', 'ms_cen_beta0', 'ms_cen_gamma0',
          'fpass_alpha0', 'fpass_beta',
          'sfr_pass_sigma', 'sfr_pass_mshift']

# For u-band
#initial = [ 0.77165, 0.53059, -8.17322, 0.01969, 11.64219, 2.51012, 0.46044, 11.97934, -1.56457, 0.49580, 1e-2, ]

# For r-band
#initial = [0.83133, 0.55393, -8.15322, 0.00146, 11.65077, 2.49197, 0.45040, 11.95805, -1.57529, 0.50116, -0.00202,]
#initial = [0.77387, 0.54726, -8.16587, 0.01481, 11.67400, 2.51272, 0.45183, 11.97927, -1.55840, 0.50351, 0.00905]

# For i-band
#initial = [0.77892, 0.54198, -8.16203, 0.01250, 11.77702, 2.52130, 0.44567, 11.97317, -1.55552, 0.50028, 0.00633,]

#std = [0.001, 0.001, 0.001,   5e-4, 2e-3, 1e-3, 1e-3, 1e-3, 1e-3,  1e-3, 2e-4]


# FIXME

pnames = ['sfr_sfms_alpha0', 'fpass_alpha0', 'fpass_beta', 'sfr_pass_mshift']
#initial = [-8.16203, 11.97317, -1.55552, 0.00633]
initial = [params[pp] for pp in pnames]
std = [5e-3, 5e-3, 5e-3, 2e-4]


pnames = ['sfr_sfms_alpha0', 'sfr_sfms_beta', 'sfr_pass_sigma', 
          'ms_cen_norm', 'ms_cen_logM1', 'ms_cen_beta0']
#initial = [-8.16203, 11.97317, -1.55552, 0.00633]
initial = [params[pp] for pp in pnames]
std = [5e-3, 5e-4, 5e-3, 
       1e-4, 1e-3, 1e-2]




######################


p0 = copy.deepcopy(params)
for p, v in zip(pnames, initial): p0[p] = v
pcur = copy.deepcopy(p0)

# Initial likelihood values
logl_rad = loglike_barlow(L_radio, logPhi_radio, errp_radio, errm_radio, pcur)
logl_opt = loglike_optical(mag_gama, Phi_gama, err_gama, band=BAND, params=pcur)
logl0 = logl_rad + logl_opt
logl = 1.*logl0

#-------------------------------------------------------------------------------

# Loop over requested number of samples
samples = []
acc = 0; rej = 0
for i in range(NSAMPLES):
    
    # Draw proposal
    pprop = copy.deepcopy(pcur)
    for p, s in zip(pnames, std):
        pprop[p] += s * np.random.normal()
    
    # Get likelihood ratio
    #logl_prop = loglike(L_data, phi_data, err_data, pprop)
    logl_prop_r = loglike_barlow(L_radio, logPhi_radio, errp_radio, errm_radio, 
                                 params=pprop)
    logl_prop_o = loglike_optical(mag_gama, Phi_gama, err_gama, band=BAND, 
                                  params=pprop)
    logl_prop = logl_prop_r + logl_prop_o
    l = np.exp(logl_prop - logl)
    
    # Perform accept/reject test
    u = np.random.uniform()
    if l >= u:
        # Accept
        print "[%3d] ACCEPT, delta(logL) = %3.3f, logL = %3.2f" % \
              (i, logl_prop - logl, logl_prop)
        if l <= 1.: print "\t *** l=%3.3f, u=%3.3f" % (l, u)
        samples.append(pprop)
        pcur = pprop
        logl = logl_prop
        acc += 1
    else:
        # Reject
        print "[%3d] REJECT, delta(logL) = %3.2f" % (i, logl_prop - logl)
        rej += 1

#-------------------------------------------------------------------------------

# Output final values and stats
print "-"*50
for p in pnames:
    print "%20s: %4.4f, %4.4f" % (p, pcur[p], p0[p])
for p in pnames: print "%5.5f," % pcur[p],
print ""
print "-"*50
print "Initial logL: %4.1f" % logl0
print "Final logL:   %4.1f" % logl
print "Delta:        %4.1f" % (logl - logl0)
print "-"*50
print "Accepted: %d / %d" % (acc, acc+rej)
print "Fraction: %3.1f %%" % (100. * float(acc) / float(acc + rej))
print "-"*50
print pcur

#-------------------------------------------------------------------------------

# Plot best-fit result (radio)
P.subplot(121)
P.plot(L_radio, radio_lumfn(L_radio, pcur), 'r-', lw=1.8)
P.plot(L_radio, radio_lumfn(L_radio, p0), 'b-', lw=1.8)

# Symmetrise radio errors
logerr_radio = 0.5 * (np.abs(errp_radio) + np.abs(errm_radio))
err_radio = Phi_radio * (10.**logerr_radio - 1.)
P.errorbar(L_radio, Phi_radio, yerr=err_radio, color='k', marker='.', ls='none')
#P.errorbar(L_data, phi_data, yerr=[errp, np.abs(errm)], color='k', marker='.')
P.yscale('log')
P.xscale('log')


# Plot best-fit result (optical)
P.subplot(122)
P.plot(mag_gama, optical_lumfn(mag_gama, BAND, pcur), 'r-', lw=1.8)
P.plot(mag_gama, optical_lumfn(mag_gama, BAND, p0), 'b-', lw=1.8)
P.errorbar(mag_gama, Phi_gama, yerr=err_gama, color='k', marker='.', ls='none')
P.gca().invert_xaxis()
#P.errorbar(L_data, phi_data, yerr=[errp, np.abs(errm)], color='k', marker='.')
P.yscale('log')
#P.xscale('log')

P.tight_layout()
P.show()
