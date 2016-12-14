/*
 * Generate mock galaxy properties using a Monte Carlo defined by the ghost 
 * model.
 * 
 *   -- Phil Bull 2016 <philbull@gmail.com>
 */

#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include <assert.h>
#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

// Define optical frequency bands
const char BAND_NAMES[5] = "ugriz";
const double BAND_WAVELENGTHS[5] = {3543., 4770., 6231., 7625., 9134.};
const int BAND_NUM = 5;

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

////////////////////////////////////////////////////////////////////////////////
// I/O utility functions
////////////////////////////////////////////////////////////////////////////////

int band_index(char band){
    /*
    Get the index of a named band.
    */
    int idx = -1;
    
    // Find index of requested band
    for(int i=0; i < BAND_NUM; i++){
        if(band == BAND_NAMES[i]){
            idx = i;
        }
    } // end i loop
    
    // Check to make sure band was found
    if(idx < 0){
        printf("ERROR: Band '%c' not found.\n", band);
        exit(1);
    }
    return idx;
}


void load_halos_from_file(char* fname, struct Catalogue *cat){
    /*
    Load catalogue of DM halos from specified file.
    */
    int success;
    char *buf = NULL;
    double temp;
    size_t len;
    FILE *f;
    
    // Count number of rows in file
    f = fopen(fname, "r");
    cat->nhalos = 0;
    while(getline(&buf, &len, f) != -1){
        (cat->nhalos)++;
    }
    printf("Found %d halos in %s.\n", cat->nhalos, fname);
    rewind(f); // Return to start of file
    
    // Allocate halo arrays
    cat->mhalo = malloc(sizeof(double) * (cat->nhalos));
    cat->z = malloc(sizeof(double) * (cat->nhalos));
    
    // Load catalogue into arrays (assumes only two columns, space-separated)
    for(int i=0; i < cat->nhalos; i++){
        // Halo mass (column 0)
        success = getdelim(&buf, &len, ' ', f);
        (cat->mhalo)[i] = strtof(buf, NULL);
        
        // Redshift (column 1)
        success = getdelim(&buf, &len, '\n', f);
        (cat->z)[i] = strtof(buf, NULL);
    }
    
    // Close file and free buffers
    free(buf);
    fclose(f);
}


void save_catalogue(char* fname, struct Catalogue *cat){
    /*
    Save generated mock catalogue to file.
    */
    FILE *f;
    int i, j;
    
    // Open file for writing
    f = fopen(fname, "w");
    
    // Loop through halos
    for(i=0; i < cat->nhalos; i++){
        fprintf(f, "%12.8e %12.8e\n", cat->mhalo[i], cat->z[i]);
    } // end i loop
    
    printf("Done writing to file.\n");
    
    // Close file
    fclose(f);
}



void default_params(struct Params *p){
    /*
    Populate parameter set with default parameter values.
    */
    p->ms_cen_logM1 = 12.498161809000001;
    p->ms_cen_norm = 0.017739115493300001;
    p->ms_cen_mu = 0.019;
    p->ms_cen_nu = -0.72;
    p->ms_cen_beta0 = 1.3211623293100001;
    p->ms_cen_beta1 = 0.17;
    p->ms_cen_gamma0 = 0.59559404090000001;
    p->ms_cen_gamma1 = -0.26;
    p->ms_cen_logM2 = 11.8;
    p->ms_cen_sigmainf = 0.030873590974500001;
    p->ms_cen_sigma1 = 0.55697362786500004;
    p->ms_cen_xi = 4.25;
    
    // Passive fraction parameters
    p->fpass_alpha0 = 10.804369278199999;
    p->fpass_alpha1 = 0.5;
    p->fpass_beta = -2.4362458372;
    p->fpass_zeta = -1.62086498436;
    
    // SFMS parameters
    p->sfr_sfms_alpha0 = -0.077393675287999994;
    p->sfr_sfms_alpha1 = 1.07;
    p->sfr_sfms_beta = 1.0373848138599999;
    p->sfr_sfms_sigma = 0.39059251672099998;
    
    // Passive sequence parameters
    p->sfr_pass_mshift = 0.00105527163519;
    p->sfr_pass_sigma = 0.028559680849299999;
    
    // Optical extinction parameters
    p->extinction_tau0 = 1.040598;
    p->extinction_beta = 5.4783395970699997;
    p->extinction_diskfac = 1.4294834244600001;
    p->extinction_kappa = 0.003534;
    p->extinction_lambda0 = 4977.189036;
    
    // Optical parameters
    p->opt_mstar_amp = 3876.55879;
    p->opt_mstar_c = -1.0108709;
    p->opt_mstar_beta = -0.000242854097;
    p->opt_cross_beta = -0.262636097;
    p->opt_cross_gamma = 0.290177366;
    p->opt_cross_amp[0] = -3.61194266;
    p->opt_cross_amp[1] = -2.70027825;
    p->opt_cross_amp[2] = -2.01635745;
    p->opt_cross_amp[3] = -1.72575487;
    p->opt_cross_amp[4] = -1.56393268;
    p->opt_offset[0] = -27.3176081;
    p->opt_offset[1] = -25.9445857;
    p->opt_offset[2] = -25.2836569;
    p->opt_offset[3] = -24.981922;
    p->opt_offset[4] = -24.8096689;
}


////////////////////////////////////////////////////////////////////////////////
// ghost model pdfs and supporting functions
////////////////////////////////////////////////////////////////////////////////

double mass_stellar_cen(double mhalo, double z, struct Params p){
    /*
    Fitting function for the central galaxy stellar mass of a given halo, 
    using Eq.2 of Moster, B. P. et al. 2010, ApJ, 710, 903.
    [Mvir should be in Msun]
    */
    double logM1, M1, norm, beta, gamma;
    
    // Redshift-dependent functions (Eqs. 23-26 of Moster et al.)
    logM1 = p.ms_cen_logM1 * pow(1. + z, p.ms_cen_mu);
    M1 = pow(10., logM1);
    norm = p.ms_cen_norm * pow(1. + z, p.ms_cen_nu);
    beta = p.ms_cen_beta0 + p.ms_cen_beta1 * z;
    gamma = p.ms_cen_gamma0 * pow(1.+z, p.ms_cen_gamma1);
    
    return 2. * mhalo * norm / (pow(mhalo/M1, -beta) + pow(mhalo/M1, gamma));
}


double pdf_mass_stellar_cen(double mhalo, double z, struct Params p, gsl_rng *rng){
    /*
    Prob. density function for stellar mass, given halo mass and redshift. Uses 
    the form from Moster et al. (2010). Central galaxies.
    */
    double M2, mean_ms, sigma;
    
    // Mean stellar mass as fn. of halo mass
    mean_ms = mass_stellar_cen(mhalo, z, p);
    
    // Scatter as a fn. of halo mass [Eq. 12 of Moster et al. (2010)]
    M2 = pow(10., p.ms_cen_logM2);
    sigma = p.ms_cen_sigmainf 
          + p.ms_cen_sigma1 * (1. - 2./M_PI*atan(p.ms_cen_xi * log10(mhalo/M2)));
    sigma *= log(10.); // sigma in dex
    
    // Ensure that sigma is +ve, and larger than the value needed for 
    // integration to converge
    if(sigma < 2e-2){ sigma = 2e-2; }
    
    // Return pdf
    return gsl_ran_lognormal(rng, 
                             log(mean_ms),
                             sigma); // FIXME: wrong args?!
    //return exp(-log(mstar/mean_ms)**2./(2.*sigma**2.)) \
    //     / (np.sqrt(2.*np.pi)*sigma*Ms);
}


double f_passive(double mstar, double z, struct Params p){
    /*
    Fraction of passive galaxies at a given stellar mass and redshift.
    */
    double c = 0.5 * (1. + tanh(p.fpass_zeta));
    return c + (1. - c) 
      / pow( 1. + (mstar / pow(10., p.fpass_alpha0 + p.fpass_alpha1*z)), p.fpass_beta);
}

bool pdf_galaxy_type(double mstar, double z, struct Params p, gsl_rng *rng){
    /*
    Draw galaxy type (passive vs. star-forming).
    */
    double fpass = f_passive(mstar, z, p);
    
    // Draw uniform random number and apply passive cut
    if(gsl_rng_uniform(rng) > fpass){
        return false; // Star-forming
    }else{
        return true; // Passive
    }
}


double sfr_sfms(double mstar, double z, struct Params p){
    /*
    Mean SFR of the star-formation main sequence.
    */
    return pow(10., p.sfr_sfms_alpha0 + p.sfr_sfms_alpha1 * z) 
         * pow(mstar/1e10, p.sfr_sfms_beta);
}


/////////////////////////////// FIXME: passive sequence


double pdf_sfr_sfms(double mstar, double z, struct Params p, gsl_rng *rng){
    /*
    Prob. density function for SFR on the SF main sequence, given stellar mass 
    and redshift, p(SFR | M_*, z).
    */
    return gsl_ran_lognormal(rng,
                             log( sfr_sfms(mstar, z, p) ), 
                             p.sfr_sfms_sigma * log(10.)); // FIXME: wrong args?
    //return np.exp(-np.log(sfr/mean_sfr)**2./(2.*sigma**2.)) \
    //     / (np.sqrt(2.*np.pi)*sigma*sfr)
}

double pdf_sfr_passive_lognormal(double mstar, double z, struct Params p, gsl_rng *rng){
    /*
    Prob. density function for SFR on the SF main sequence, given stellar mass 
    and redshift, p(SFR | M_*, z). [log-normal version]
    */
    // Take the SFMS powerlaw, shift it by some factor, and change scatter
    
    // Sanity check on shift parameter
    assert(p.sfr_pass_mshift >= 0.);
    
    // Draw log-normal realisation
    gsl_ran_lognormal(rng,
                      log( sfr_sfms(mstar, z, p) * p.sfr_pass_mshift ),
                      p.sfr_pass_sigma * log(10.)); // sigma in dex
    //return np.exp(-0.5 * (np.log(sfr/mean_sfr) / sigma)**2.) \
    //     / (np.sqrt(2.*np.pi)*sigma*sfr)
}

double tau_extinction(double sintheta, double mstar, char band, double z, 
                     struct Params p){
    /*
    Dust extinction optical depth as a function of inclination angle.
    */
    double l;
    
    // Get band wavelength
    l = BAND_WAVELENGTHS[band_index(band)];
    
    // Calculate optical depth and return
    return p.extinction_tau0 * pow(mstar / 1e11, p.extinction_beta)
            * (1. + p.extinction_diskfac * sintheta)
            * exp(-p.extinction_kappa * (l - p.extinction_lambda0));
}

double optical_mag(double sfr, double mstar, char band, double z, struct Params p){
    /*
    Return the predicted optical magnitude given the stellar mass and SFR. 
    Calculated using an ansatz with best-fit parameters calibrated against 
    the Guo et al. catalogue.
    */
    int i;
    double mag;
    
    // Figure out which band to use
    i = band_index(band);
    
    // Ansatz, roughly taking into account 2 sources of stellar light: star 
    // formation (SFR as proxy) and older stars (via stellar mass)
    mag = p.opt_mstar_amp * ( p.opt_mstar_c 
                            + pow(mstar/1e9, p.opt_mstar_beta) ) 
        + p.opt_cross_amp[i] * pow(mstar/1e9, p.opt_cross_beta)
                             * pow(sfr, p.opt_cross_gamma) 
        - p.opt_offset[i];
    return mag;
}

////////////////////////////////////////////////////////////////////////////////
// Model realisation code
////////////////////////////////////////////////////////////////////////////////

void realise_catalogue(struct Catalogue *cat, struct Params p, gsl_rng *rng){
    /*
    Populate a mock galaxy catalogue by drawing values for all parameters.
    */
    
    // Loop over halos; populate with galaxy prooperties by traversing model
    #pragma omp parallel for
    for(int i=0; i<cat->nhalos; i++){
        
        // Stellar mass
        cat->mstar[i] = pdf_mass_stellar_cen(cat->mhalo[i], cat->z[i], p, rng);
        
        // Galaxy type
        cat->passive[i] = pdf_galaxy_type(cat->mhalo[i], cat->z[i], p, rng);
        
        // SFR (depending on galaxy type)
        if(cat->passive[i]){
            // Passive type
            cat->sfr[i] = pdf_sfr_passive_lognormal(cat->mstar[i], cat->z[i], p, rng);
        }else{
            // Star-forming type
            cat->sfr[i] = pdf_sfr_sfms(cat->mstar[i], cat->z[i], p, rng);
        }
        
        // Intrinsic optical magnitudes
        
        
    } // end i loop over halos
    
}

////////////////////////////////////////////////////////////////////////////////
// Main function
////////////////////////////////////////////////////////////////////////////////


void main(int argc, const char* argv[]){
    /*
     * 
     */
    bool *passive;
    double **catalogue;
    struct Params p;
    struct Catalogue cat;
    gsl_rng *rng;
    
    // Initialise RNG
    gsl_rng_env_setup();
    rng = gsl_rng_alloc( gsl_rng_default );
    
    // Set default values of model parameters
    default_params(&p);
    
    // Load halo catalogue
    // malloc: mhalo, z
    load_halos_from_file("test.dat", &cat);
    
    // Allocate arrays for catalogue variables
    cat.mstar = (double*)malloc(sizeof(double) * (cat.nhalos));
    cat.sfr = (double*)malloc(sizeof(double) * (cat.nhalos));
    cat.passive = (bool*)malloc(sizeof(bool) * (cat.nhalos));
    
    // Realise all variables in catalogue
    realise_catalogue(&cat, p, rng);
    
    // Output catalogue to file
    save_catalogue("mock.dat", &cat);
    
    // Free halo catalogue arrays
    free(cat.mhalo);
    free(cat.z);
    free(cat.mstar);
    free(cat.sfr);
    free(cat.passive);
    
    // Free GSL RNG objects
    gsl_rng_free(rng);
    
    return;
}
