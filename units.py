# Constants in various units

# Fundamental constants
C = 3e5 # In km/s
PI = 3.14159265359

# Distances and unit conversions
MPC_IN_CM = 3.085680e24 # 1 Mpc in cm
RAD_TO_MIN = 180./PI*60.

# SZ thermal constants
COMPTON_SCALE = 1.301854e-27 # sigma_T / m_e c^2, (keV^-1 cm^2)
X_SCALE = 4.79924e-2 # x = X_SCALE * nu [Ghz] / T [k]
T_CMB = 2.72548 # CMB temperature today, from Fixsen, ApJ 707 (2009).
Y_SCALE = 4.017105e-3 # Y_SCALE = COMPTON_SCALE * MPC_IN_CM
NU_SCALE = 1. / 56.85 # [Ghz^-1]: h nu / k_B T_CMB = nu / 56.85 Ghz

# WMAP band centre frequencies, in GHz
NU_WMAP_K =  22.
NU_WMAP_KA = 30.
NU_WMAP_Q =  40.
NU_WMAP_V =  60.
NU_WMAP_W =  90.

# Default cosmology parameters
DEFAULT_COSMO = {
 'omega_M_0': 		0.3,
 'omega_lambda_0': 	0.7,
 'omega_b_0': 		0.045,
 'omega_n_0':		0.0,
 'omega_k_0':		0.0,
 'N_nu':			0,
 'h':				0.7,
 'n':				0.96,
 'sigma_8':			0.8,
 'baryonic_effects': True
}
