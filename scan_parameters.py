#!/usr/bin/python
"""
Plot optical luminosity function as a function of parameters.
"""
import numpy as np
import pylab as P
import matplotlib.cm as cm
import galaxy_model as g
import likelihoods as like
import copy, time

BAND = 'r'

# Set model parameters
params = {'sfr_sfms_sigma': 0.39597, 'mass_mstar_min': 6.0, 'opt_bands': ['u', 'g', 'r', 'i', 'z'], 'fpass_alpha0': 10.81015, 'fpass_alpha1': 0.5, 'sfr_pass_pthres': 0.05, 'fpass_zeta': -0.92929, 'ms_cen_nu': -0.72, 'mass_mstar_max': 14.0, 'opt_pdf_sigma': [0.28122634653850337, 0.25232918833346546, 0.2468073941298409, 0.25273681573440887, 0.2724513351999828], 'extinction_alpha': -7.57886, 'extinction_beta': 5.32513, 'sfr_pass_beta': 0.37, 'extinction_diskfac': 11.25638, 'mass_mhalo_max': 16.0, 'ms_cen_sigma1': 0.55174, 'opt_offset': [-27.3176081, -25.9445857, -25.2836569, -24.981922, -24.8096689], 'ms_cen_logM1': 12.49492, 'mhi_omegaHI': 0.0001, 'ms_cen_mu': 0.019, 'nsamp_mstar': 200, 'ms_cen_norm': 0.01591, 'nsamp_mhalo': 200, 'opt_cross_beta': -0.262636097, 'ms_cen_gamma0': 0.59821, 'ms_cen_gamma1': -0.26, 'extinction_amp': 0.32755, 'mass_mhalo_min': 5.0, 'ms_cen_xi': 4.25, 'opt_cross_gamma': 0.290177366, 'opt_mstar_amp': 3876.55879, 'ms_cen_logM2': 11.8, 'opt_mstar_beta': -0.000242854097, 'sfr_pass_type': 'shifted sfms', 'sfr_min': -5.0, 'sfr_pass_mshift': 0.00735, 'ms_cen_sigmainf': 0.03174, 'sfr_pass_alpha1': 0.0, 'sfr_sfms_gamma': 1.9, 'nsamp_sfr': 200, 'sfr_sfms_mscale': 1.0, 'sfr_sfms_beta': 1.03109, 'opt_cross_amp': [-3.61194266, -2.70027825, -2.01635745, -1.72575487, -1.56393268], 'sfr_sfms_alpha1': 1.07, 'sfr_sfms_alpha0': -0.05094, 'opt_pdf_mean': [-0.06587481168919591, -0.053777765775930214, -0.01854712885192855, -0.008538656095465969, -0.008732300503716532], 'sfr_pass_mscale': 1.0, 'sfr_pass_sigma': 0.16627, 'ms_cen_beta0': 1.25341, 'ms_cen_beta1': 0.17, 'opt_mstar_c': -1.0108709, 'sfr_max': 4.0, 'fpass_beta': -2.64471, 'sfr_pass_gamma': 1.9, 'sfr_pass_sfrmin': 1e-07, 'sfr_pass_alpha0': -5.0, 'mhi_vc1': 200.0, 'mhi_vc0': 50.0}

# Load my halo model
pkfile = "camb_pk_z0.dat"
hm = g.HaloModel(pkfile, h=0.67, om=0.32)

# Load GAMA luminosity function
mag_gama, Phi_gama, err_gama = like.load_gama_lf(band=BAND, h=0.67)
mag_gama = mag_gama[:-3]
Phi_gama = Phi_gama[:-3]
err_gama = err_gama[:-3]

def loglike_optical_lf(mag, phi, err, band, params, z, mphi):
    """
    Calculate log-likelihood for optical luminosity function.
    """
    #mphi = optical_lumfn(mag, band, params, z=z)
    logl = -0.5 * np.sum( (phi - mphi)**2. / err**2. )
    return logl

#-------------------------------------------------------------------------------
#P.subplot(111)

"""
#mag = np.linspace(-25., -17., 30)
pname1 = 'extinction_beta'
vals1 = np.linspace(3., 8., 10)

pname2 = 'extinction_amp'
vals2 = np.logspace(-6., -2., 10)

l = np.zeros((vals1.size, vals2.size))
for i in range(vals1.size):
    for j in range(vals2.size):
        print i, j
        params[pname1] = vals1[i]
        params[pname2] = vals2[j]
        
        magfn_sfms, magfn_pass = g.optical_mag_fn_atten(hm, mag_gama, band=BAND, 
                                                        z=0., params=params)
        logL = loglike_optical_lf(mag_gama, Phi_gama, err_gama, BAND, params, 0.,
                                  magfn_sfms + magfn_pass)
        l[i,j] = -logL

P.matshow(l, extent=[np.min(np.log10(vals2)), np.max(np.log10(vals2)),
                     np.max(vals1), np.min(vals1)])
P.colorbar()
P.ylabel(pname1)
P.xlabel(pname2)
"""

pname = 'extinction_amp'
p0 = params[pname]
#vals = params['extinction_amp'] * np.linspace(0.3, 1.5, 10)
vals = params['extinction_amp'] * np.logspace(-5., -2., 10)

P.errorbar(mag_gama, Phi_gama, yerr=err_gama, marker='.', color='r', 
           ls='none', capsize=4., elinewidth=1.5, mew=1.5)

#params['extinction_amp'] = 1e-5
for v in vals:
    print v / p0
    params[pname] = v
    magfn_sfms, magfn_pass = g.optical_mag_fn_atten(hm, mag_gama, band=BAND, 
                                                    z=0., params=params)
    logL = loglike_optical_lf(mag_gama, Phi_gama, err_gama, BAND, params, 0.,
                                  magfn_sfms + magfn_pass)
    
    P.plot(mag_gama, magfn_sfms + magfn_pass, lw=1.8, 
           label="%s = %3.3e, $\log \mathcal{L} = %2.1f$" % (pname, v/p0, -logL))

P.title(BAND)
P.legend(loc='lower left', frameon=False)
P.yscale('log')
P.gca().invert_xaxis()
P.tight_layout()


P.show()
