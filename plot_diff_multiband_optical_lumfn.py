#!/usr/bin/python
"""
Plot difference between best-fit luminosity function and data for all optical 
bands, vs. the GAMA data.
"""
import numpy as np
import pylab as P
import matplotlib.gridspec as gridspec
import galaxy_model as g
import likelihoods as like
import copy
import time
import scipy.interpolate

# Set model parameters
params = {'sfr_sfms_sigma': 0.4, 'mass_mstar_min': 6.0, 'opt_bands': ['u', 'g', 'r', 'i', 'z'], 'fpass_alpha0': 10.8, 'fpass_alpha1': 0.5, 'sfr_pass_pthres': 0.05, 'fpass_zeta': -0.3, 'ms_cen_nu': -0.72, 'mass_mstar_max': 14.0, 'opt_pdf_sigma': [0.28122634653850337, 0.25232918833346546, 0.2468073941298409, 0.25273681573440887, 0.2724513351999828], 'sfr_pass_beta': 0.37, 'mass_mhalo_max': 16.0, 'ms_cen_sigma1': 0.05, 'opt_offset': [-27.3176081, -25.9445857, -25.2836569, -24.981922, -24.8096689], 'opt_cross_amp': [-3.61194266, -2.70027825, -2.01635745, -1.72575487, -1.56393268], 'opt_mstar_beta': -0.000242854097, 'mhi_omegaHI': 0.0001, 'ms_cen_mu': 0.019, 'nsamp_mstar': 200, 'ms_cen_norm': 0.02, 'opt_cross_beta': -0.262636097, 'ms_cen_gamma0': 0.65, 'ms_cen_gamma1': -0.26, 'mass_mhalo_min': 5.0, 'ms_cen_xi': 4.25, 'opt_cross_gamma': 0.290177366, 'opt_mstar_amp': 3876.55879, 'ms_cen_logM2': 11.8, 'ms_cen_logM1': 12.1, 'sfr_pass_gamma': 1.9, 'sfr_min': -5.0, 'sfr_pass_mshift': 0.03, 'ms_cen_sigmainf': 0.16, 'sfr_pass_sfrmin': 1e-07, 'sfr_sfms_gamma': 1.9, 'sfr_pass_mscale': 1.0, 'sfr_sfms_mscale': 1.0, 'sfr_sfms_beta': 0.9, 'nsamp_mhalo': 200, 'sfr_sfms_alpha1': 1.07, 'sfr_sfms_alpha0': 0.005, 'opt_pdf_mean': [-0.06587481168919591, -0.053777765775930214, -0.01854712885192855, -0.008538656095465969, -0.008732300503716532], 'nsamp_sfr': 200, 'sfr_pass_sigma': 0.4, 'ms_cen_beta0': 1.4, 'ms_cen_beta1': 0.17, 'opt_mstar_c': -1.0108709, 'sfr_max': 4.0, 'fpass_beta': -1.1, 'sfr_pass_type': 'shifted sfms', 'sfr_pass_alpha1': 0.0, 'sfr_pass_alpha0': -5.0, 'mhi_vc1': 200.0, 'mhi_vc0': 50.0}


#-------------------------------------------------------------------------------
# Calculate luminosity function from my model

# Load my halo model
pkfile = "camb_pk_z0.dat"
hm = g.HaloModel(pkfile, h=0.67, om=0.32)

# Number density as a function of optical magnitude
mag = np.linspace(-27., -13., 30) #100)
#obsmag = np.linspace(-24., -13., 10)

# Bands and colour-coding
bands = ['u', 'g', 'r', 'i', 'z']
colours = ['b', 'g', 'r', 'y', 'm']
colours = ['#DA49DA', '#00a5ff', '#ff8200', '#F34334', '#9D2020'] # Freq. -> colour, but more contrasty

#-------------------------------------------------------------------------------
# Plot optical luminosity function as a fn. of magnitude
ax = []
gs = gridspec.GridSpec(2, 3)
ax.append(P.subplot(gs[0, 0]))
ax.append(P.subplot(gs[0, 1]))
ax.append(P.subplot(gs[0, 2]))
ax.append(P.subplot(gs[1, 0]))
ax.append(P.subplot(gs[1, 1]))
ax.append(P.subplot(gs[1, 2]))
#gs.update(left=0.05, right=0.48, wspace=0.05)

for i in range(len(bands)):
    b = bands[i]
    col = colours[i]
    
    print "%s-band" % b
    
    # Load GAMA band binned luminosity fun. for this band
    gama_mag, gama_n, gama_err = like.load_gama_lf(band=b, h=0.67)
    
    # Calculate number density as a function of magnitude
    dndmag_sfms_dust, dndmag_pass_dust, dndmag_sfms, dndmag_pass = \
        g.optical_mag_fn_dust( hm, gama_mag, mag, band=b, z=0.,
                               params=params, include_intrinsic=True)

    # Results from my calculation
    #P.plot(mag, dndmag_sfms + dndmag_pass, color=col, lw=1.8, 
    #       label="%s-band" % b)
    dndmag_th = scipy.interpolate.interp1d(mag, dndmag_sfms + dndmag_pass, 
                                           kind='linear', bounds_error=False)
    
    #ax[i].errorbar(gama_mag, gama_n - dndmag_th(gama_mag), yerr=gama_err, 
    #           marker='.', color=col, ls='none')
    ax[i].errorbar(gama_mag, gama_n/dndmag_th(gama_mag) - 1., yerr=gama_err/dndmag_th(gama_mag), 
                   marker='.', color=col, ls='none')
    ax[i].axhline(0., color='k', lw=1.4, alpha=0.5)
    
    # Plot settings
    #P.yscale('log')    
    ax[i].set_xlim((-26.5, -13.))
    #P.ylim((1e-7, 1e-1))
    ax[i].invert_xaxis()

    ax[i].tick_params(axis='both', which='major', labelsize=14, size=8., 
                        width=1.5, pad=8.)
    ax[i].tick_params(axis='both', which='minor', labelsize=14, size=5., 
                        width=1.5, pad=8.)
    #P.grid()
    #P.legend(loc='upper right', frameon=False)

    ax[i].set_ylabel(r'$dn/dm$ $[{\rm Mpc}^{-3}]$', fontsize=14.)
    ax[i].set_xlabel('mag', fontsize=14.)


#-------------------------------------------------------------------------------
# Plot radio luminosity function as a fn. of log-luminosity
print "Radio 1.4 GHz"

# Load 6dFGS data from Mauch & Sadler (astro-ph/0612018)
L_rad, Phi_rad, errp_rad, errm_rad = \
                                like.load_mauch_lf(h=0.67, starburst_corr=False)
L_rad2, Phi_rad2, errp_rad2, errm_rad2 = \
                                like.load_mauch_lf(h=0.67, starburst_corr=True)

# Symmetrise radio errors
logerr_rad = 0.5 * (np.abs(errp_rad) + np.abs(errm_rad))
err_rad = Phi_rad * (10.**logerr_rad - 1.)
erp = Phi_rad * (10.**errp_rad - 1.)
erm = Phi_rad * (1. - 10.**errm_rad)
logerr_rad2 = 0.5 * (np.abs(errp_rad2) + np.abs(errm_rad2))
err_rad2 = Phi_rad2 * (10.**logerr_rad2 - 1.)

def radio_lumfn(L, _params):
    """
    Calculate radio luminosity function for a given set of model parameters.
    """
    print _params
    # Number density as a function of sfr, dn/dlog(sfr)
    sfr = L * 5.52e-29 # erg/s/Hz, Bell (2003), Eq. 6
    dndlogsfr_sfms, dndlogsfr_pass = g.sfr_fn(hm, sfr, z=0., params=_params)
    phi = dndlogsfr_sfms + dndlogsfr_pass
    return phi

#P.plot(L_rad, radio_lumfn(L_rad, params), 'k-', lw=1.8, label="Radio 1.4 GHz SF")
#ax[-1].errorbar(L_rad, Phi_rad - radio_lumfn(L_rad, params), 
#           yerr=err_rad, color='k', marker='.', ls='none')

#ax[-1].errorbar(L_rad, Phi_rad - radio_lumfn(L_rad, params), 
#           yerr=[erm, erp], color='k', marker='.', ls='none')
rlumfn = radio_lumfn(L_rad, params)
ax[-1].errorbar(L_rad, Phi_rad/rlumfn - 1., 
                yerr=[erm/rlumfn, erp/rlumfn], color='k', marker='.', ls='none')

ax[-1].axhline(0., color='k', lw=1.4, alpha=0.5)
ax[-1].set_xscale('log')
#P.ylim((1e-9, 1e-2))

ax[-1].tick_params(axis='both', which='major', labelsize=14, size=8., 
                    width=1.5, pad=8.)
ax[-1].tick_params(axis='both', which='minor', labelsize=14, size=5., 
                    width=1.5, pad=8.)
#P.legend(loc='upper right', frameon=False)

ax[-1].set_ylabel(r'$dn/d\log L$ $[{\rm Mpc}^{-3}]$', fontsize=14.)
ax[-1].set_xlabel('L [erg/s/Hz]', fontsize=14.)

#P.tight_layout()
P.show()
