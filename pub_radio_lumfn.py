#!/usr/bin/python
"""
Plot luminosity function for radio band.
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

# Get current best-fit model
params = params_bf

# Load my halo model
pkfile = "camb_pk_z0.dat"
hm = g.HaloModel(pkfile, h=0.67, om=0.32)

#-------------------------------------------------------------------------------
# Set-up gridspec
P.figure(figsize=(8,8))
gs = gridspec.GridSpec(2,1)
gs.update(wspace=0.0, hspace=0.0)

#-------------------------------------------------------------------------------

# Load 6dFGS local radio LF data from Mauch & Sadler (astro-ph/0612018)
L_radio, Phi_radio, errp_radio, errm_radio = \
               like.load_mauch_lf(fname="../lumfns/lumfunc_6dfgs.dat", h=0.67)
logPhi_radio = np.log10(Phi_radio)
erp = Phi_radio * (10.**errp_radio - 1.)
erm = Phi_radio * (1. - 10.**errm_radio)

L = np.logspace(np.log10(np.min(L_radio))-1.,
                np.log10(np.max(L_radio))+1.,
                30)

def radio_lumfn(L, _params):
    """
    Calculate radio luminosity function for a given set of model parameters.
    """
    # Number density as a function of sfr, dn/dlog(sfr)
    sfr = L * 5.52e-29 # erg/s/Hz, Bell (2003), Eq. 6
    dndlogsfr_sfms, dndlogsfr_pass = g.sfr_fn(hm, sfr, z=0., params=_params)
    #phi = dndlogsfr_sfms #+ dndlogsfr_pass
    return dndlogsfr_sfms, dndlogsfr_pass

ax1 = P.subplot(gs[0])
ax2 = P.subplot(gs[1])

#L_radx = np.logspace(23., 31., 25)
radlum_sfms, radlum_pass = radio_lumfn(L, params)
ax1.plot(L, radlum_sfms, 'k-', lw=1.8, label="Radio 1.4 GHz SF")
#ax1.plot(L, radlum_sfms, 'k--', lw=1.8)
#ax1.plot(L, radlum_pass, 'k', lw=1.8, dashes=[2,2])

ax1.errorbar(L_radio, Phi_radio, yerr=[erm, erp], color='k',
           marker='.', ls='none', capsize=4., elinewidth=1.5, mew=1.5)

# Fractional difference (only SF galaxies were used in the fit)
radlum_tot = radlum_sfms #+ radlum_pass

# Residuals (obtained by interpolating in (y-)log space)
logdn_interp = scipy.interpolate.interp1d(L, 
                           np.log(radlum_sfms), kind='linear',
                           bounds_error=False)
n_pred = np.exp(logdn_interp(L_radio))
fracdev = (Phi_radio / n_pred) - 1.
fracdev[np.where(fracdev == -1.)] = np.nan # Discard zero values

ax2.errorbar(L_radio, fracdev, yerr=[erm/n_pred, erp/n_pred], 
             color='k', marker='.', ls='none', capsize=4., elinewidth=1.5, mew=1.5)
ax2.axhline(0., color='k', ls='dashed', lw=1.8)

# Axis tick labels
for ax in [ax1, ax2]:
    ax.set_xscale('log')
    ax.set_xlim((3e26, 8e30))

    ax.tick_params(axis='both', which='major', labelsize=20, size=8., 
                        width=1.5, pad=8.)
    ax.tick_params(axis='both', which='minor', labelsize=20, size=5., 
                        width=1.5, pad=8.)

ax1.tick_params(axis='x', which='major', labelbottom='off')
ax2.set_ylim((-1., 1.3))

ax1.set_yscale('log')
ax1.set_ylim((5e-8, 3e-2))
ax1.set_ylabel(r'$dn/d\log L$ $[{\rm Mpc}^{-3}]$', fontsize=18.)

ax2.set_ylabel(r'${\rm Frac.}\, {\rm Diff.}$', fontsize=18., labelpad=-3.)
ax2.set_xlabel(r'$L\, [{\rm erg/s/Hz}]$', fontsize=18.)

P.tight_layout()
P.savefig("../draft/lumfn_radio.pdf")
P.show()
