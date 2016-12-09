#!/usr/bin/python
"""
Output best-fit optical magnitude relation parameters in LaTeX table format.
"""
import numpy as np
from bestfit import *

params = params_bf

names = {
    'extinction_kappa':     r'$\kappa$',
    'extinction_tau0':      r'$\tau_0$',
    'extinction_beta':      r'$\beta_\tau$',
    'extinction_lambda0':   r'$\lambda_0$',
    'extinction_diskfac':   r'$a_{\rm disk}$',
    'opt_pdf_sigma':        r'$\sigma_m^{(\nu)}$',
    'opt_offset':           r'$c_0^{(\nu)}$',
    'opt_cross_beta':       r'$\beta_\times$',
    'opt_cross_gamma':      r'$\gamma_\times$',
    'opt_mstar_amp':        r'$A_\star$',
    'opt_mstar_beta':       r'$\beta_\star$',
    'opt_cross_amp':        r'$A_\times^{(\nu)}$',
    'opt_pdf_mean':         r'$\mu^{(\nu)}$',
    'opt_mstar_c':          r'$c_\star$',
}

order = ['opt_mstar_amp', 'opt_mstar_beta', 'opt_mstar_c',
         'opt_cross_amp', 'opt_cross_beta', 'opt_cross_gamma',
         'opt_offset', 'opt_pdf_mean', 'opt_pdf_sigma',
         'extinction_tau0', 'extinction_lambda0',
         'extinction_kappa', 'extinction_diskfac',
         'extinction_beta']
#namelist = names.keys()
#namelist.sort()

for n in order:
    print "%s &" % names[n],
    if type(params[n]) == list:
        print " & ".join( ["%3.5f" % p for p in params[n]] ),
        print "\\\\"
    else:
        print "\multicolumn{5}{|c|}{%3.3f} \\\\" % params[n]

