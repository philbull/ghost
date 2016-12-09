#!/usr/bin/python
"""
Find best-fit parameters from a given chain.
"""
import numpy as np
import sys
from bestfit import *

if len(sys.argv) > 1:
    fname = sys.argv[1]
    print "Loading %s" % fname
else:
    print "Filename needed."
    sys.exit()

#params0 = {'sfr_sfms_sigma': 0.38750005196600001, 'mass_mstar_min': 6.0, 'opt_bands': ['u', 'g', 'r', 'i', 'z'], 'fpass_alpha0': 10.8205970498, 'fpass_alpha1': 0.5, 'sfr_pass_pthres': 0.05, 'fpass_zeta': -1.5353087936700001, 'ms_cen_nu': -0.72, 'mass_mstar_max': 14.0, 'opt_pdf_sigma': [0.28122634653850337, 0.25232918833346546, 0.2468073941298409, 0.25273681573440887, 0.2724513351999828], 'extinction_alpha': -7.57886, 'extinction_beta': 5.4223162433100001, 'sfr_pass_beta': 0.37, 'extinction_diskfac': 13.516079320399999, 'mass_mhalo_max': 16.0, 'ms_cen_sigma1': 0.60802361084000001, 'opt_offset': [-27.3176081, -25.9445857, -25.2836569, -24.981922, -24.8096689], 'ms_cen_logM1': 12.489042828500001, 'mhi_omegaHI': 0.0001, 'ms_cen_mu': 0.019, 'nsamp_mstar': 300, 'ms_cen_norm': 0.015799159819799999, 'nsamp_mhalo': 200, 'opt_cross_beta': -0.262636097, 'ms_cen_gamma0': 0.54792407909100005, 'ms_cen_gamma1': -0.26, 'extinction_amp': 0.29403960120099998, 'mass_mhalo_min': 7.0, 'ms_cen_xi': 4.25, 'opt_cross_gamma': 0.290177366, 'opt_mstar_amp': 3876.55879, 'ms_cen_logM2': 11.8, 'opt_mstar_beta': -0.000242854097, 'sfr_pass_type': 'shifted sfms', 'sfr_min': -9.0, 'sfr_pass_mshift': 0.0053486148026400003, 'ms_cen_sigmainf': -0.031605451195699998, 'sfr_pass_alpha1': 0.0, 'sfr_sfms_gamma': 1.9, 'nsamp_sfr': 250, 'sfr_sfms_mscale': 1.0, 'sfr_sfms_beta': 1.0910253563500001, 'opt_cross_amp': [-3.61194266, -2.70027825, -2.01635745, -1.72575487, -1.56393268], 'sfr_sfms_alpha1': 1.07, 'sfr_sfms_alpha0': -0.10162165209600001, 'opt_pdf_mean': [-0.06587481168919591, -0.053777765775930214, -0.01854712885192855, -0.008538656095465969, -0.008732300503716532], 'sfr_pass_mscale': 1.0, 'sfr_pass_sigma': 0.012386348129800001, 'ms_cen_beta0': 1.29481129595, 'ms_cen_beta1': 0.17, 'opt_mstar_c': -1.0108709, 'sfr_max': 4.0, 'fpass_beta': -2.7709240453000001, 'sfr_pass_gamma': 1.9, 'sfr_pass_sfrmin': 1e-07, 'sfr_pass_alpha0': -5.0, 'mhi_vc1': 200.0, 'mhi_vc0': 50.0}

params0 = params_bf

def load_samples(fname):
    """
    Load samples from emcee MCMC chain. Loads column names from the header.
    """
    # Load data
    dat = np.genfromtxt(fname).T
    
    # Load names from header
    f = open(fname, 'r')
    # Removes leading #, trailing \n and splits on spaces
    names = f.readline()[:-1].split(" ")[1:]
    f.close()
    
    # Repack into dictionary
    chain = {}
    for i in range(len(names)):
        chain[names[i]] = dat[i]
    return chain

# Load MCMC samples
chain = load_samples(fname)

# Find maximum likelihood value
idx = np.where(chain['logl'] == np.max(chain['logl']))[0][0]
print "-"*50
print "\tlogL_max = %3.2f" % chain['logl'][idx]
print "-"*50
lbls = chain.keys()
print lbls
lbls.sort()
for key in lbls:
    if key in params0:
        print "%16s: %3.5f" % (key, chain[key][idx])
        params0[key] = chain[key][idx]
print "-"*50
print params0
