
#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

// Define optical frequency bands
//const char BAND_NAMES[5] = "ugriz";
//const double BAND_WAVELENGTHS[5] = {3543., 4770., 6231., 7625., 9134.};
//const int BAND_NUM = 5;

struct Catalogue{
    double *mhalo;
    double *z;
    double *mstar;
    double *sfr;
    bool *passive;
    int nhalos;
};

// Parameters of the model
struct Params{
    
    // Stellar mass-halo mass relation for centrals
    double ms_cen_logM1;
    double ms_cen_norm;
    double ms_cen_mu;
    double ms_cen_nu;
    double ms_cen_beta0;
    double ms_cen_beta1;
    double ms_cen_gamma0;
    double ms_cen_gamma1;
    double ms_cen_logM2;
    double ms_cen_sigmainf;
    double ms_cen_sigma1;
    double ms_cen_xi;
    
    // Passive fraction parameters
    double fpass_alpha0;
    double fpass_alpha1;
    double fpass_beta;
    double fpass_zeta;
    
    // SFMS parameters
    double sfr_sfms_alpha0;
    double sfr_sfms_alpha1;
    double sfr_sfms_beta;
    double sfr_sfms_sigma;
    
    // Passive sequence parameters
    double sfr_pass_mshift;
    double sfr_pass_sigma;
    
    // Optical extinction parameters
    double extinction_tau0;
    double extinction_beta;
    double extinction_diskfac;
    double extinction_kappa;
    double extinction_lambda0;
    
    // Optical parameters
    double opt_mstar_amp;
    double opt_mstar_c;
    double opt_mstar_beta;
    double opt_cross_amp[5];
    double opt_cross_beta;
    double opt_cross_gamma;
    double opt_offset[5];
    
};

// I/O utility functions
int band_index(char band);
void load_halos_from_file(char* fname, struct Catalogue *cat);
void save_catalogue(char* fname, struct Catalogue *cat);
void default_params(struct Params *p);

// ghost model pdfs and supporting functions
double mass_stellar_cen(double mhalo, double z, struct Params p);
double pdf_mass_stellar_cen(double mhalo, double z, struct Params p, gsl_rng *rng);
double f_passive(double mstar, double z, struct Params p);
bool pdf_galaxy_type(double mstar, double z, struct Params p, gsl_rng *rng);
double sfr_sfms(double mstar, double z, struct Params p);
double pdf_sfr_sfms(double mstar, double z, struct Params p, gsl_rng *rng);
double pdf_sfr_passive_lognormal(double mstar, double z, struct Params p, gsl_rng *rng);
double tau_extinction(double sintheta, double mstar, char band, double z, struct Params p);
double optical_mag(double sfr, double mstar, char band, double z, struct Params p);

// Model realisation code
void realise_catalogue(struct Catalogue *cat, struct Params p, gsl_rng *rng);

// Main function
int main(int argc, const char* argv[]);
