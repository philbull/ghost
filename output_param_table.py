#!/usr/bin/python
"""
Output best-fit optical magnitude relation parameters in LaTeX table format.
"""
import numpy as np
from bestfit import *

params = params_bf


names = {
    'fpass_alpha0':         r'$\alpha_f$',
    'fpass_beta':           r'$\beta_f$',
    'fpass_zeta':           r'$\zeta_f$',
    
    'ms_cen_norm':          r'$A_\star$',
    'ms_cen_beta0':         r'$\beta_\star$',
    'ms_cen_gamma0':        r'$\gamma_\star$',
    'ms_cen_logM1':         r'$\log_{10} M^\star_1$',
    'ms_cen_logM2':         r'$\log_{10} M^\star_2$',
    'ms_cen_mu':            r'$\mu_\star$',
    'ms_cen_nu':            r'$\nu_\star$',
    'ms_cen_sigma1':        r'$\sigma_1^\star$',
    'ms_cen_sigmainf':      r'$\sigma_\infty^\star$',
    'ms_cen_xi':            r'$\xi_\star$',
    
    'sfr_pass_alpha0':      r'$\alpha_{\rm pass}$',
    'sfr_pass_beta':        r'$\beta_{\rm pass}$',
    'sfr_pass_gamma':       r'$\gamma_{\rm pass}$',
#    'sfr_pass_mscale':      r'$$',
    'sfr_pass_mshift':      r'$a_{\rm pass}$',
#    'sfr_pass_pthres':      r'$$',
#    'sfr_pass_sfrmin':      r'$$',
    'sfr_pass_sigma':       r'$\sigma_{\rm pass}$',
    
    'sfr_sfms_alpha0':      r'$\alpha_{\rm SFMS}$',
    'sfr_sfms_beta':        r'$\beta_{\rm SFMS}$',
    'sfr_sfms_gamma':       r'$\gamma_{\rm SFMS}$',
    'sfr_sfms_sigma':       r'$\sigma_{\rm SFMS}$',
}

namelist = names.keys()
namelist.sort()

for n in namelist:
    print "%s & & & %3.3f \\\\" % (names[n], params[n])

