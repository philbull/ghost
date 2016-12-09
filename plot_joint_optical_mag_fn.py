#!/usr/bin/python
"""
Plot joint luminosity function for two optical bands.
"""
import numpy as np
import pylab as P
import matplotlib.cm as cm
import matplotlib
import galaxy_model as g
import likelihoods as like
import copy, time
from bestfit import *

params = params_bf

# Load my halo model
pkfile = "camb_pk_z0.dat"
hm = g.HaloModel(pkfile, h=0.67, om=0.32)

#-------------------------------------------------------------------------------
#P.subplot(111)

g_r = np.linspace(-0.2, 1.5, 20)[::-1]
mag_r = np.linspace(-24., -18., 25)

magfn = np.zeros((g_r.size, mag_r.size))
for i in range(g_r.size):
    for j in range(mag_r.size):
        _r = mag_r[j]
        _g = g_r[i] + _r
        magfn_sfms, magfn_pass = g.joint_optical_mag_fn_atten(hm, _g, _r, 
                                      band1='g', band2='r', z=0., params=params)
        magfn[i,j] = magfn_sfms + magfn_pass


#P.matshow(np.log10(magfn), aspect=5.,
#          extent=[np.min(mag_r), np.max(mag_r), np.min(g_r), np.max(g_r)])

G_R, R = np.meshgrid(g_r, mag_r)

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
cplt1 = P.contour(G_R, R, np.log10(magfn.T), levels=[-8., -6., -4., -3., -2.],
                  linewidths=2.5, colors='r')

"""
zz = np.log10(magfn_sfms.flatten())
zz[np.where(np.isinf(zz))] = -10.
print np.min(zz), np.max(zz)
#P.tricontourf(g_r.flatten(), mr.flatten(), zz)
P.tripcolor(g_r.flatten(), mr.flatten(), zz)
"""

#P.colorbar()
#P.tight_layout()
P.show()
