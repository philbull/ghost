#!/usr/bin/python
"""
Plot optical magnitude as a function of SFR and stellar mass.
"""
import numpy as np
import pylab as P
import matplotlib
import galaxy_model as g
import likelihoods as like
from bestfit import *
import time

BAND = 'g'

bands = ['u', 'g', 'r', 'i', 'z']
colours = ['#DA49DA', '#00a5ff', '#ff8200', '#F34334', '#9D2020'] # Freq. -> colour, but more contrasty
COL = colours[bands.index(BAND)]

# Get best-fit parameters
params = params_bf

sfr = np.logspace(-4., 3., 100)
mstar = np.logspace(6., 11., 101)
SFR, MSTAR = np.meshgrid(sfr, mstar)

# Calculate mean optical magnitude
mag = g.optical_mag(SFR, MSTAR, BAND, z=0., params=params)

# Plot optical magnitude as a fn. of SFR and stellar mass
P.subplot(111)

matplotlib.rcParams['contour.negative_linestyle'] = 'solid'
cplt1 = P.contour(MSTAR, SFR, mag, 
                  levels=np.arange(-25., -12., 1),
                  linewidths=2.5, colors=COL)
P.clabel(cplt1, fmt='%d', fontsize=16)

P.xscale('log')
P.yscale('log')

P.grid()

P.tight_layout()
P.show()
