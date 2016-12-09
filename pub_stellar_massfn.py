#!/usr/bin/python
"""
Plot stellar mass function.
"""
import numpy as np
import pylab as P
import galaxy_model as g
import likelihoods as like
import copy, time
import matplotlib.ticker
import matplotlib.gridspec as gridspec
import scipy.interpolate
from bestfit import *

COLOUR_SFMS = '#0088FF'
COLOUR_PASS = '#E72327'

# Set model parameters
#params = {'sfr_sfms_sigma': 0.39597, 'mass_mstar_min': 6.0, 'opt_bands': ['u', 'g', 'r', 'i', 'z'], 'fpass_alpha0': 10.81015, 'fpass_alpha1': 0.5, 'sfr_pass_pthres': 0.05, 'fpass_zeta': -0.92929, 'ms_cen_nu': -0.72, 'mass_mstar_max': 14.0, 'opt_pdf_sigma': [0.28122634653850337, 0.25232918833346546, 0.2468073941298409, 0.25273681573440887, 0.2724513351999828], 'extinction_alpha': -7.57886, 'extinction_beta': 5.32513, 'sfr_pass_beta': 0.37, 'extinction_diskfac': 11.25638, 'mass_mhalo_max': 16.0, 'ms_cen_sigma1': 0.55174, 'opt_offset': [-27.3176081, -25.9445857, -25.2836569, -24.981922, -24.8096689], 'ms_cen_logM1': 12.49492, 'mhi_omegaHI': 0.0001, 'ms_cen_mu': 0.019, 'nsamp_mstar': 200, 'ms_cen_norm': 0.01591, 'nsamp_mhalo': 200, 'opt_cross_beta': -0.262636097, 'ms_cen_gamma0': 0.59821, 'ms_cen_gamma1': -0.26, 'extinction_amp': 0.32755, 'mass_mhalo_min': 5.0, 'ms_cen_xi': 4.25, 'opt_cross_gamma': 0.290177366, 'opt_mstar_amp': 3876.55879, 'ms_cen_logM2': 11.8, 'opt_mstar_beta': -0.000242854097, 'sfr_pass_type': 'shifted sfms', 'sfr_min': -5.0, 'sfr_pass_mshift': 0.00735, 'ms_cen_sigmainf': 0.03174, 'sfr_pass_alpha1': 0.0, 'sfr_sfms_gamma': 1.9, 'nsamp_sfr': 200, 'sfr_sfms_mscale': 1.0, 'sfr_sfms_beta': 1.03109, 'opt_cross_amp': [-3.61194266, -2.70027825, -2.01635745, -1.72575487, -1.56393268], 'sfr_sfms_alpha1': 1.07, 'sfr_sfms_alpha0': -0.05094, 'opt_pdf_mean': [-0.06587481168919591, -0.053777765775930214, -0.01854712885192855, -0.008538656095465969, -0.008732300503716532], 'sfr_pass_mscale': 1.0, 'sfr_pass_sigma': 0.16627, 'ms_cen_beta0': 1.25341, 'ms_cen_beta1': 0.17, 'opt_mstar_c': -1.0108709, 'sfr_max': 4.0, 'fpass_beta': -2.64471, 'sfr_pass_gamma': 1.9, 'sfr_pass_sfrmin': 1e-07, 'sfr_pass_alpha0': -5.0, 'mhi_vc1': 200.0, 'mhi_vc0': 50.0}


# Best-fit: optical g,z, radio, no SMF
#params = {'sfr_sfms_sigma': 0.415595676119, 'mass_mstar_min': 6.0, 'opt_bands': ['u', 'g', 'r', 'i', 'z'], 'fpass_alpha0': 10.839733046599999, 'fpass_alpha1': 0.5, 'sfr_pass_pthres': 0.05, 'fpass_zeta': -0.95027169466200001, 'ms_cen_nu': -0.72, 'mass_mstar_max': 14.0, 'opt_pdf_sigma': [0.28122634653850337, 0.25232918833346546, 0.2468073941298409, 0.25273681573440887, 0.2724513351999828], 'extinction_alpha': -7.57886, 'extinction_beta': 5.3450337472999996, 'sfr_pass_beta': 0.37, 'extinction_diskfac': 10.2412022264, 'mass_mhalo_max': 16.0, 'ms_cen_sigma1': 0.56640755541800003, 'opt_offset': [-27.3176081, -25.9445857, -25.2836569, -24.981922, -24.8096689], 'opt_cross_amp': [-3.61194266, -2.70027825, -2.01635745, -1.72575487, -1.56393268], 'opt_mstar_beta': -0.000242854097, 'mhi_omegaHI': 0.0001, 'ms_cen_mu': 0.019, 'nsamp_mstar': 300, 'ms_cen_norm': 0.015185988130400001, 'opt_cross_beta': -0.262636097, 'ms_cen_gamma0': 0.56119468698499997, 'ms_cen_gamma1': -0.26, 'extinction_amp': 0.33240776116999998, 'mass_mhalo_min': 5.0, 'ms_cen_xi': 4.25, 'opt_cross_gamma': 0.290177366, 'opt_mstar_amp': 3876.55879, 'ms_cen_logM2': 11.8, 'ms_cen_logM1': 12.488922480999999, 'sfr_pass_gamma': 1.9, 'sfr_min': -5.0, 'sfr_pass_mshift': 0.0083967790705399992, 'ms_cen_sigmainf': 0.025222009908500001, 'sfr_pass_sfrmin': 1e-07, 'sfr_sfms_gamma': 1.9, 'sfr_pass_mscale': 1.0, 'sfr_sfms_mscale': 1.0, 'sfr_sfms_beta': 0.99195297218300005, 'nsamp_mhalo': 300, 'sfr_sfms_alpha1': 1.07, 'sfr_sfms_alpha0': -0.0581345697711, 'opt_pdf_mean': [-0.06587481168919591, -0.053777765775930214, -0.01854712885192855, -0.008538656095465969, -0.008732300503716532], 'nsamp_sfr': 300, 'sfr_pass_sigma': 0.011843956765499999, 'ms_cen_beta0': 1.2756497875999999, 'ms_cen_beta1': 0.17, 'opt_mstar_c': -1.0108709, 'sfr_max': 4.0, 'fpass_beta': -2.7139265055099999, 'sfr_pass_type': 'shifted sfms', 'sfr_pass_alpha1': 0.0, 'sfr_pass_alpha0': -5.0, 'mhi_vc1': 200.0, 'mhi_vc0': 50.0}

# Best-fit: optical g,z, radio, SMF
#params = {'sfr_sfms_sigma': 0.095346928085199997, 'mass_mstar_min': 6.0, 'opt_bands': ['u', 'g', 'r', 'i', 'z'], 'fpass_alpha0': 10.9257522857, 'fpass_alpha1': 0.5, 'sfr_pass_pthres': 0.05, 'fpass_zeta': -0.356976178955, 'ms_cen_nu': -0.72, 'mass_mstar_max': 14.0, 'opt_pdf_sigma': [0.28122634653850337, 0.25232918833346546, 0.2468073941298409, 0.25273681573440887, 0.2724513351999828], 'extinction_alpha': -7.57886, 'extinction_beta': 6.3819123521099996, 'sfr_pass_beta': 0.37, 'extinction_diskfac': -1.0095804421400001, 'mass_mhalo_max': 16.0, 'ms_cen_sigma1': 0.35055806168300002, 'opt_offset': [-27.3176081, -25.9445857, -25.2836569, -24.981922, -24.8096689], 'opt_cross_amp': [-3.61194266, -2.70027825, -2.01635745, -1.72575487, -1.56393268], 'opt_mstar_beta': -0.000242854097, 'mhi_omegaHI': 0.0001, 'ms_cen_mu': 0.019, 'nsamp_mstar': 200, 'ms_cen_norm': 0.021171742042100001, 'opt_cross_beta': -0.262636097, 'ms_cen_gamma0': 0.607698300016, 'ms_cen_gamma1': -0.26, 'extinction_amp': 2.2483719206999999, 'mass_mhalo_min': 5.0, 'ms_cen_xi': 4.25, 'opt_cross_gamma': 0.290177366, 'opt_mstar_amp': 3876.55879, 'ms_cen_logM2': 11.8, 'ms_cen_logM1': 12.0862669462, 'sfr_pass_gamma': 1.9, 'sfr_min': -5.0, 'sfr_pass_mshift': 0.0013389284153300001, 'ms_cen_sigmainf': 0.023304494212900002, 'sfr_pass_sfrmin': 1e-07, 'sfr_sfms_gamma': 1.9, 'sfr_pass_mscale': 1.0, 'sfr_sfms_mscale': 1.0, 'sfr_sfms_beta': 1.5871068530100001, 'nsamp_mhalo': 200, 'sfr_sfms_alpha1': 1.07, 'sfr_sfms_alpha0': -0.455230306503, 'opt_pdf_mean': [-0.06587481168919591, -0.053777765775930214, -0.01854712885192855, -0.008538656095465969, -0.008732300503716532], 'nsamp_sfr': 200, 'sfr_pass_sigma': 0.63368804654999999, 'ms_cen_beta0': 1.0826299856799999, 'ms_cen_beta1': 0.17, 'opt_mstar_c': -1.0108709, 'sfr_max': 4.0, 'fpass_beta': -1.0535586695200001, 'sfr_pass_type': 'shifted sfms', 'sfr_pass_alpha1': 0.0, 'sfr_pass_alpha0': -5.0, 'mhi_vc1': 200.0, 'mhi_vc0': 50.0}

# Get current best-fit model
params = params_bf


MSTAR_MIN = None
HUBBLE = 0.67

# Load my halo model
pkfile = "camb_pk_z0.dat"
hm = g.HaloModel(pkfile, h=0.67, om=0.32)

#-------------------------------------------------------------------------------
# Set-up gridspec
P.figure(figsize=(8,8))
gs = gridspec.GridSpec(2,1)
gs.update(wspace=0.0, hspace=0.0)

#-------------------------------------------------------------------------------

# Load SDSS-GALEX stellar mass function
mstar_sdss, sf_phi_sdss, sf_errp_sdss, sf_errm_sdss, \
            qu_phi_sdss, qu_errp_sdss, qu_errm_sdss \
                = like.load_sdss_smf(h=HUBBLE, mstar_min=MSTAR_MIN, 
                                     convert_errors=True)
# Trim low-mass datapoints
#mstar_sdss, sf_phi_sdss, sf_errp_sdss, sf_errm_sdss, \
#qu_phi_sdss, qu_errp_sdss, qu_errm_sdss \
#  = [var[4:] for var in (mstar_sdss, sf_phi_sdss, sf_errp_sdss, sf_errm_sdss,
#                         qu_phi_sdss, qu_errp_sdss, qu_errm_sdss)]

sf_logphi_sdss = np.log10(sf_phi_sdss)
qu_logphi_sdss = np.log10(qu_phi_sdss)

ax1 = P.subplot(gs[0])
ax2 = P.subplot(gs[1])

mstar = np.logspace(7.5, 13.5, 100)
dndlogms = g.stellar_mass_fn(hm, mstar, z=0., params=params)
fpass = g.f_passive(mstar, z=0., params=params)

ax1.plot(mstar, (1. - fpass) * dndlogms, color=COLOUR_SFMS, lw=1.8, 
         label="Star-forming")
ax1.plot(mstar, fpass * dndlogms, color=COLOUR_PASS, ls='solid', lw=1.8, 
         label="Passive")



ax1.errorbar(mstar_sdss, sf_phi_sdss, yerr=[sf_errm_sdss, sf_errp_sdss], 
             color=COLOUR_SFMS, marker='.', ls='none', capsize=4., 
             elinewidth=1.5, mew=1.5)

ax1.errorbar(mstar_sdss, qu_phi_sdss, yerr=[qu_errm_sdss, qu_errp_sdss], 
             color=COLOUR_PASS, marker='.', ls='none', capsize=4., 
             elinewidth=1.5, mew=1.5)

ax1.legend(loc='upper right', frameon=False)

# Fractional difference
# Residuals (obtained by interpolating in (y-)log space)
sfms_interp = scipy.interpolate.interp1d(mstar, 
                           np.log((1. - fpass) * dndlogms), kind='linear',
                           bounds_error=False)
pass_interp = scipy.interpolate.interp1d(mstar, 
                           np.log(fpass * dndlogms), kind='linear',
                           bounds_error=False)

n_pred_sfms = np.exp(sfms_interp(mstar_sdss))
n_pred_pass = np.exp(pass_interp(mstar_sdss))

fracdev_sfms = (sf_phi_sdss / n_pred_sfms) - 1.
fracdev_pass = (qu_phi_sdss / n_pred_pass) - 1.
fracdev_sfms[np.where(fracdev_sfms == -1.)] = np.nan # Discard zero values
fracdev_pass[np.where(fracdev_pass == -1.)] = np.nan # Discard zero values

ax2.errorbar(mstar_sdss, fracdev_sfms, 
             yerr=[sf_errm_sdss/n_pred_sfms, sf_errp_sdss/n_pred_sfms], 
             color=COLOUR_SFMS, marker='.', ls='none', capsize=4., 
             elinewidth=1.5, mew=1.5)
             
ax2.errorbar(mstar_sdss, fracdev_pass, 
             yerr=[qu_errm_sdss/n_pred_pass, qu_errp_sdss/n_pred_pass], 
             color=COLOUR_PASS, marker='.', ls='none', capsize=4., 
             elinewidth=1.5, mew=1.5)
ax2.axhline(0., color='k', ls='dashed', lw=1.8)

# Axis tick labels
for ax in [ax1, ax2]:
    ax.set_xscale('log')
    ax.set_xlim((1e8, 5e12))

    ax.tick_params(axis='both', which='major', labelsize=20, size=8., 
                        width=1.5, pad=8.)
    ax.tick_params(axis='both', which='minor', labelsize=20, size=5., 
                        width=1.5, pad=8.)

ax1.tick_params(axis='x', which='major', labelbottom='off')
ax2.set_ylim((-1., 1.8))

ax1.set_yscale('log')
ax1.set_ylim((5e-9, 5e-2))
ax1.set_ylabel(r'$dn/d\log M_\star$ $[{\rm Mpc}^{-3}]$', fontsize=18.)

ax2.set_ylabel(r'${\rm Frac.}\, {\rm Diff.}$', fontsize=18., labelpad=-3.)
ax2.set_xlabel(r'$M_\star\, [M_\odot]$', fontsize=18.)

P.tight_layout()
#P.savefig("../draft/smf.pdf")
P.show()
