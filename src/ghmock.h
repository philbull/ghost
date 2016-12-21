
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
//const float BAND_WAVELENGTHS[5] = {3543., 4770., 6231., 7625., 9134.};
//const int BAND_NUM = 5;

struct Catalogue{
    float *mhalo;
    float *z;
    float *mstar;
    float *sfr;
    bool *passive;
    int nhalos;
};

// Define optical frequncy bands
#define NUM_BANDS 5
enum {BAND_U, BAND_G, BAND_R, BAND_I, BAND_Z};
extern const float BAND_WAVELENGTHS[NUM_BANDS];
extern const char BAND_NAMES[NUM_BANDS];

// Parameters of the model
struct Params{
    
    // Stellar mass-halo mass relation for centrals
    float ms_cen_logM1;
    float ms_cen_norm;
    float ms_cen_mu;
    float ms_cen_nu;
    float ms_cen_beta0;
    float ms_cen_beta1;
    float ms_cen_gamma0;
    float ms_cen_gamma1;
    float ms_cen_logM2;
    float ms_cen_sigmainf;
    float ms_cen_sigma1;
    float ms_cen_xi;
    
    // Passive fraction parameters
    float fpass_alpha0;
    float fpass_alpha1;
    float fpass_beta;
    float fpass_zeta;
    
    // SFMS parameters
    float sfr_sfms_alpha0;
    float sfr_sfms_alpha1;
    float sfr_sfms_beta;
    float sfr_sfms_sigma;
    
    // Passive sequence parameters
    float sfr_pass_mshift;
    float sfr_pass_sigma;
    
    // Optical extinction parameters
    float extinction_tau0;
    float extinction_beta;
    float extinction_diskfac;
    float extinction_kappa;
    float extinction_lambda0;
    
    // Optical parameters
    float opt_mstar_amp;
    float opt_mstar_c;
    float opt_mstar_beta;
    float opt_cross_amp[NUM_BANDS];
    float opt_cross_beta;
    float opt_cross_gamma;
    float opt_offset[NUM_BANDS];
    float opt_pdf_sigma[NUM_BANDS]; 
    float opt_pdf_mean[NUM_BANDS];
    
};

// I/O utility functions
int band_index(char band);
void load_halos_from_file(char* fname, struct Catalogue *cat);
void save_catalogue(char* fname, struct Catalogue *cat);
void default_params(struct Params *p);

// ghost model pdfs and supporting functions
float mass_stellar_cen(float mhalo, float z, struct Params p);
float draw_mass_stellar_cen(float mhalo, float z, struct Params p, gsl_rng *rng);
float f_passive(float mstar, float z, struct Params p);
bool draw_galaxy_type(float mstar, float z, struct Params p, gsl_rng *rng);
float sfr_sfms(float mstar, float z, struct Params p);
float draw_sfr_sfms(float mstar, float z, struct Params p, gsl_rng *rng);
float draw_sfr_passive_lognormal(float mstar, float z, struct Params p, gsl_rng *rng);
float tau_extinction(float sintheta, float mstar, char band, float z, struct Params p);
float optical_mag(float sfr, float mstar, char band, float z, struct Params p);
float draw_optical_mag_intrinsic(float sfr, float mstar, char band, float z, 
                                  struct Params p, gsl_rng *rng);
float draw_optical_mag_atten(float mag_int, float mstar, char band, float z, 
                              struct Params p, gsl_rng *rng);

// Model realisation code
void realise_physical_properties(struct Catalogue *cat, struct Params p, gsl_rng *rng);

// Main function
int main(int argc, const char* argv[]);
