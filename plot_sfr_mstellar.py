#!/usr/bin/python
"""
Plot pdf in the stellar mass-SFR plane.
"""
import numpy as np
import pylab as P
import galaxy_model as g
import matplotlib

# Set model parameters
params = {'sfr_sfms_sigma': 0.39597, 'mass_mstar_min': 6.0, 'opt_bands': ['u', 'g', 'r', 'i', 'z'], 'fpass_alpha0': 10.81015, 'fpass_alpha1': 0.5, 'sfr_pass_pthres': 0.05, 'fpass_zeta': -0.92929, 'ms_cen_nu': -0.72, 'mass_mstar_max': 14.0, 'opt_pdf_sigma': [0.28122634653850337, 0.25232918833346546, 0.2468073941298409, 0.25273681573440887, 0.2724513351999828], 'extinction_alpha': -7.57886, 'extinction_beta': 5.32513, 'sfr_pass_beta': 0.37, 'extinction_diskfac': 11.25638, 'mass_mhalo_max': 16.0, 'ms_cen_sigma1': 0.55174, 'opt_offset': [-27.3176081, -25.9445857, -25.2836569, -24.981922, -24.8096689], 'ms_cen_logM1': 12.49492, 'mhi_omegaHI': 0.0001, 'ms_cen_mu': 0.019, 'nsamp_mstar': 200, 'ms_cen_norm': 0.01591, 'nsamp_mhalo': 200, 'opt_cross_beta': -0.262636097, 'ms_cen_gamma0': 0.59821, 'ms_cen_gamma1': -0.26, 'extinction_amp': 0.32755, 'mass_mhalo_min': 5.0, 'ms_cen_xi': 4.25, 'opt_cross_gamma': 0.290177366, 'opt_mstar_amp': 3876.55879, 'ms_cen_logM2': 11.8, 'opt_mstar_beta': -0.000242854097, 'sfr_pass_type': 'shifted sfms', 'sfr_min': -5.0, 'sfr_pass_mshift': 0.00735, 'ms_cen_sigmainf': 0.03174, 'sfr_pass_alpha1': 0.0, 'sfr_sfms_gamma': 1.9, 'nsamp_sfr': 200, 'sfr_sfms_mscale': 1.0, 'sfr_sfms_beta': 1.03109, 'opt_cross_amp': [-3.61194266, -2.70027825, -2.01635745, -1.72575487, -1.56393268], 'sfr_sfms_alpha1': 1.07, 'sfr_sfms_alpha0': -0.05094, 'opt_pdf_mean': [-0.06587481168919591, -0.053777765775930214, -0.01854712885192855, -0.008538656095465969, -0.008732300503716532], 'sfr_pass_mscale': 1.0, 'sfr_pass_sigma': 0.16627, 'ms_cen_beta0': 1.25341, 'ms_cen_beta1': 0.17, 'opt_mstar_c': -1.0108709, 'sfr_max': 4.0, 'fpass_beta': -2.64471, 'sfr_pass_gamma': 1.9, 'sfr_pass_sfrmin': 1e-07, 'sfr_pass_alpha0': -5.0, 'mhi_vc1': 200.0, 'mhi_vc0': 50.0}

ALPHA1 = 0.5
ALPHA2 = 0.5

# Load my halo model
pkfile = "camb_pk_z0.dat"
hm = g.HaloModel(pkfile, h=0.67, om=0.32)

"""
# Plot f_pass
ms = np.logspace(7., 14., 400)
fpass = g.f_passive(ms, z=0., params=params)
P.plot(ms, fpass, 'k-', lw=1.8)
P.xscale('log')
P.show()
"""

"""
# Number density as a function of halo mass, dn/dlog(mhalo)
mhalo = np.logspace(8., 18., 500)
dndlogm = hm.dndlogm(mhalo, z=0., params=params)

# Number density as a function of stellar mass, dn/dlog(mstar)
mstar = np.logspace(5., 18., 500)
dndlogms = g.stellar_mass_fn(hm, mstar, z=0., params=params)

# Number density as a function of sfr, dn/dlog(sfr)
sfr = np.logspace(-5., 4., 200)
dndlogsfr_sfms, dndlogsfr_pass = g.sfr_fn(hm, sfr, z=0., params=params)
"""

# 2D grid: SFR and Mstar
##sfr = np.logspace(-7., 3., 200)
##mstar = np.logspace(7., 12., 500)
mstar = np.logspace(7., 13., 121) #61)
sfr = np.logspace(-7., 4., 80) #40)
n_sf_sfms, n_sf_pass = g.sfr_mstar_plane(hm, sfr, mstar, z=0., params=params)

# FIXME: To compare with Lagos, should normalise along Mstar direction
extent = [np.log10(np.min(mstar)), np.log10(np.max(mstar)),
          np.log10(np.min(sfr)), np.log10(np.max(sfr))]
#plt = P.imshow(np.log10(n_sf_sfms), origin='lower', cmap='Reds', extent=extent, 
#          vmin=-8., vmax=-2., aspect=1./2., interpolation='none', alpha=ALPHA1)
#plt = P.imshow(np.log10(n_sf_pass), origin='lower', cmap='Blues', extent=extent, 
#          vmin=-8., vmax=-2., aspect=1./2., interpolation='none', alpha=ALPHA1)

print np.min(n_sf_sfms), np.max(n_sf_sfms)

# Load from De Lucia catalogue at snapshot=63 (z=0)
#dl_mvir, dl_mstellar, dl_sfr = np.genfromtxt(
#                             "../delucia_mass/sfr_mstellar_mhalo_direct.sql",
#                             delimiter=',', skip_header=1).T

# Guo (2010)
dl_mvir, dl_mstellar, dl_sfr, mag_u, mag_g, mag_r, mag_i, mag_z = \
    np.genfromtxt('../delucia_mass/guo_colours_z0.dat', 
    delimiter=',', skip_header=1).T


dl_mstellar *= 1e10 / 0.73 # convert to Msun
dl_mvir *= 1e10 / 0.73 # convert to Msun

# 2D histogram of De Lucia sample, gives dN/dlogM*/dlogSFR
mm = np.logspace(np.log10(np.min(mstar)), np.log10(np.max(mstar)), 60)
ss = np.logspace(np.log10(np.min(sfr)), np.log10(np.max(sfr)), 39)
#mm = np.logspace(8.5, 12., 60)
#ss = np.logspace(-3., 3.5, 39)
n_dl, xx, yy = np.histogram2d(np.log10(dl_mstellar), np.log10(dl_sfr), 
               bins=[np.log10(mm), np.log10(ss)])
Lbox = 62.5 / 0.73 # Mpc/h -> Mpc, box size
n_dl = n_dl.T / Lbox**3. # (dN/dlogM*/dlogSFR)/dV = dn/dlogM*/dlogSFR

#P.plot(np.log10(dl_mstellar), np.log10(dl_sfr), 'r,')
#P.imshow(np.log10(n_dl), origin='lower', cmap='Greens', extent=extent, 
#          aspect=1./2., interpolation='none', alpha=ALPHA2,
#          vmin=-8., vmax=-2.)

# Plot contours
matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
MM, SS = np.meshgrid(np.log10(mstar), np.log10(sfr))
cplt1 = P.contour(MM, SS, np.log10(n_sf_sfms), levels=[-8., -6., -4., -3., -2.],
                  linewidths=1.8, colors='r')
cplt2 = P.contour(MM, SS, np.log10(n_sf_pass), levels=[-8., -6., -4., -3., -2.],
                  linewidths=1.8, colors='b')

xxc = 0.5 * (xx[1:] + xx[:-1])
yyc = 0.5 * (yy[1:] + yy[:-1])
XX, YY = np.meshgrid(xxc, yyc)
cplt3 = P.contour(XX, YY, np.log10(n_dl), levels=[-8., -6.,-5., -4., -3., -2.],
                  linewidths=1.8, colors='g')

P.clabel(cplt1, fmt='$10^{%d}$', fontsize=16)
P.clabel(cplt2, fmt='$10^{%d}$', fontsize=16)
P.clabel(cplt3, fmt='$10^{%d}$', fontsize=16)


#-------------------------------------------------------------------------------
# Plot best-fit powerlaw curves from various sources
f_sfr = lambda ms, alpha, beta: np.log10(10.**alpha * (ms)**beta)
logms = np.log10(mstar)

#P.plot(logms, f_sfr(mstar, -9., 0.85), 'r-', lw=1.5, label="emcee lower")
#P.plot(logms, f_sfr(mstar, -12., 1.15), 'r--', lw=1.5, label="emcee upper")
P.plot(logms, f_sfr(mstar, -3.14, 0.37), 'b-', lw=1.5, label="Wang fixed beta")
#P.plot(logms, f_sfr(mstar, -4.65, 0.5), 'b--', lw=1.5, label="Wang by-eye fit")

#P.plot(logms, f_sfr(mstar, -9.72, 0.957), 'g-', lw=1.5, label="new emcee sfms")
#P.plot(logms, f_sfr(mstar, -18.84, 1.256), 'g--', lw=1.5, label="new emcee pass")

#P.plot(logms, 0.65 * (logms-8.5) - 1.5, 'r-', lw=2.) # Moustakas rough fit to dividing line
#P.axvline(np.log10(8e9), color='r', lw=2., ls='dashed')

#P.plot(logms, f_sfr(mstar, -9.2, 0.96), 'g-', lw=1.5, label="Lagos Bow06.KS by-eye")
#P.plot(logms, f_sfr(mstar, -18.0, 1.64), 'y-', lw=1.5, label="Newfit sfms")
#P.plot(logms, f_sfr(mstar, -18.0 + np.log10(4.51), 1.64), 'y--', lw=1.5, label="Newfit pass")

#P.plot(logms, f_sfr(mstar, -4.5, 0.45), 'm-', lw=1.5, label="Testfit")

P.legend(loc='lower right', frameon=False)

#-------------------------------------------------------------------------------

P.xlim((np.min(np.log10(mstar)), np.max(np.log10(mstar))))
P.ylim((np.min(np.log10(sfr)), np.max(np.log10(sfr))))
P.xlabel(r"$\log_{10} M_*$ $[M_\odot]$", fontsize=18)
P.ylabel(r"$\log_{10} \psi$ $[M_\odot/{\rm yr}]$", fontsize=18)
#P.colorbar()

P.gca().tick_params(axis='both', which='major', labelsize=20, size=8., 
                    width=1.5, pad=8.)
P.gca().tick_params(axis='both', which='minor', labelsize=20, size=5., 
                    width=1.5, pad=8.)

P.tight_layout()
P.show()
