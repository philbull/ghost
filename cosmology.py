from units import *
import numpy as np
import scipy.integrate

class Cosmology(object):
    
    def __init__(self, cosmo):
        """
        Precompute cosmological functions.
        """
        self.update_cosmology(cosmo)
    
    def update_cosmology(self, cosmo):
        """
        Set cosmological parameter dictionary and precompute cosmo functions
        """
        self.cosmo = cosmo
        self.H, self.r = self.background_evolution_splines(cosmo)
        self.E = lambda z: self.H(z) / (100. * self.cosmo['h'])
    
    def rho_m(self, z):
        """
        Matter density as a function of redshift (in Msun/Mpc^3).
        """
        h = self.cosmo['h']
        om = self.cosmo['omega_M_0']
        return 2.7756e11 * h**2. * om * (1. + z)**3.
    
    def background_evolution_splines(self, cosmo, zmax=15., nsamples=600):
        """
        Calculate interpolation functions for background functions of redshift:
          - H(z), Hubble rate in km/s/Mpc
          - r(z), comoving distance in Mpc
        """
        _z = np.linspace(0., zmax, nsamples)
        a = 1. / (1. + _z)
        H0 = (100.*cosmo['h']); w0 = cosmo['w0']; wa = cosmo['wa']
        om = cosmo['omega_M_0']; ol = cosmo['omega_lambda_0']
        ok = 1. - om - ol
        
        # Sample Hubble rate H(z) and comoving dist. r(z) at discrete points
        omegaDE = ol * np.exp(3.*wa*(a - 1.)) / a**(3.*(1. + w0 + wa))
        E = np.sqrt( om * a**(-3.) + ok * a**(-2.) + omegaDE )
        _H = H0 * E
        
        r_c = np.concatenate( ([0.], scipy.integrate.cumtrapz(1./E, _z)) )
        if ok > 0.:
            _r = C/(H0*np.sqrt(ok)) * np.sinh(r_c * np.sqrt(ok))
        elif ok < 0.:
            _r = C/(H0*np.sqrt(-ok)) * np.sin(r_c * np.sqrt(-ok))
        else:
            _r = (C/H0) * r_c
        
        # Construct interpolating functions and return
        r = scipy.interpolate.interp1d(_z, _r, kind='linear', bounds_error=False)
        H = scipy.interpolate.interp1d(_z, _H, kind='linear', bounds_error=False)
        return H, r
