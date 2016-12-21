
//#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "numpy/arrayobject.h"
#include "ghmock.h"

//const char BAND_NAMES[NUM_BANDS] = "ugriz";

void set_params(PyObject *params, struct Params *p){
    /*
    Convert a Python parameter dictionary to a C struct.
    */
    PyObject *key, *val;
    
    //----------------------------------------
    // Preprocessor code
    
    // C preprocessor definition to set value in Params for each item found 
    // in parameter dictionary
    #define set_param_from_dict(NAME)            \
        key = PyString_FromString(""#NAME"");    \
        if (PyDict_Contains(params, key)){       \
            val = PyDict_GetItem(params, key);   \
            p->NAME = PyFloat_AsDouble(val);     \
        }
    
    // Specify which parameters to look for in dict
    set_param_from_dict( ms_cen_logM1 );
    set_param_from_dict( ms_cen_norm );
    set_param_from_dict( ms_cen_mu );
    set_param_from_dict( ms_cen_nu );
    set_param_from_dict( ms_cen_beta0 );
    set_param_from_dict( ms_cen_beta1 );
    set_param_from_dict( ms_cen_gamma0 );
    set_param_from_dict( ms_cen_gamma1 );
    set_param_from_dict( ms_cen_logM2 );
    set_param_from_dict( ms_cen_sigmainf );
    set_param_from_dict( ms_cen_sigma1 );
    set_param_from_dict( ms_cen_xi );
    set_param_from_dict( fpass_alpha0 );
    set_param_from_dict( fpass_alpha1 );
    set_param_from_dict( fpass_beta );
    set_param_from_dict( fpass_zeta );
    set_param_from_dict( sfr_sfms_alpha0 );
    set_param_from_dict( sfr_sfms_alpha1 );
    set_param_from_dict( sfr_sfms_beta );
    set_param_from_dict( sfr_sfms_sigma );
    set_param_from_dict( sfr_pass_mshift );
    set_param_from_dict( sfr_pass_sigma );
    set_param_from_dict( extinction_tau0 );
    set_param_from_dict( extinction_beta );
    set_param_from_dict( extinction_diskfac );
    set_param_from_dict( extinction_kappa );
    set_param_from_dict( extinction_lambda0 );
    set_param_from_dict( opt_mstar_amp );
    set_param_from_dict( opt_mstar_c );
    set_param_from_dict( opt_mstar_beta );
    set_param_from_dict( opt_cross_beta );
    set_param_from_dict( opt_cross_gamma );
    // End preprocessor code
    // FIXME: Need to set params for array-like variables!
    // FIXME: Needs opt_pdf params
    //----------------------------------------
    
    // Raise warning if error occurred
    if(PyErr_Occurred()){
        PyErr_SetString(PyExc_ValueError, 
            "set_params() failed: a value in 'params' probably couldn't be cast to the right type.");
        PyErr_Print();
        return;
    }
}

static char docstring_add_physical_properties[] = 
  "Create a realisation of galaxy physical properties on top of an input halo\n"
  "catalogue.\n\n"
  "Parameters\n"
  "----------\n"
  "mhalo, z : array_like (must have dtype==np.float32)\n"
  "  Arrays containing the halo mass [M_sun] and redshift of the objects in \n"
  "  the halo catalogue.\n\n"
  "params : dict (optional)\n"
  "  Dictionary of ghost model parameters. Uses default values for parameters\n"
  "  that are not specified. Uses all default values if argument not passed.\n\n"
  "Returns\n"
  "-------\n"
  "mstar, sfr, passive : array_like (float, float, bool)\n"
  "  Realisations of the stellar mass [M_sun], star-formation rate [M_sun/yr]\n"
  "  and whether the galaxy is passive (True) or star-forming (False).";

static PyObject* add_physical_properties(PyObject *self, PyObject *args){
    /*
    Create a realisation of galaxy physical properties on top of an input halo 
    catalogue.
    */
    
    // Initialise RNG
    gsl_rng *rng;
    gsl_rng_env_setup();
    rng = gsl_rng_alloc( gsl_rng_default );
    
    // Halo catalogue input arrays
    PyObject *arg_mhalo, *arg_z, *arg_params;
    
    // arg_params is optional, so initialise as empty dictionary first
    arg_params = PyDict_New();
    
    // Get halo catalogue arrays from input arguments
    if (!PyArg_ParseTuple(args, "OO|O", &arg_mhalo, &arg_z, &arg_params)){
        PyErr_SetString(PyExc_RuntimeError, "Failed to interpret input arguments.");
        return NULL;
    }
    
    // Populate parameter struct with default parameter values
    struct Params p;
    default_params(&p);
    
    // Set parameters defined in input parameter dict (if any)
    set_params(arg_params, &p);
    
    // Construct Catalogue structure to manage variables
    struct Catalogue cat;
    
    // Convert input arguments to numpy arrays and expose C pointers to data
    PyArrayObject *np_mhalo = (PyArrayObject*)PyArray_FROM_O(arg_mhalo);
    PyArrayObject *np_z = (PyArrayObject*)PyArray_FROM_O(arg_z);
    
    // Test input data type; must be NPY_FLOAT32
    PyArray_Descr *dtype_mhalo = PyArray_DTYPE(np_mhalo);
    PyArray_Descr *dtype_z = PyArray_DTYPE(np_z);
    
    if (dtype_mhalo->type_num != NPY_FLOAT32){
        PyErr_SetString( PyExc_RuntimeError, 
                         "Data type of input 'mhalo' not allowed. "
                         "Must be np.float32.");
        return NULL;
    }
    if (dtype_z->type_num != NPY_FLOAT32){
        PyErr_SetString( PyExc_RuntimeError, 
                         "Data type of input 'z' not allowed. "
                         "Must be np.float32.");
        return NULL;
    }
    int N = (int)PyArray_DIM(np_mhalo, 0); // Get length of input arrays
    
    // Create new ndarrays and provide data access pointers for C code
    int ndim = 1;
    npy_intp shape[1] = {N};
    PyObject *np_mstar = PyArray_SimpleNew(ndim, shape, NPY_FLOAT);
    PyObject *np_sfr = PyArray_SimpleNew(ndim, shape, NPY_FLOAT);
    PyObject *np_passive = PyArray_SimpleNew(ndim, shape, NPY_BOOL);
    if ((np_mstar == NULL) || (np_sfr == NULL) || (np_passive == NULL)) {
        PyErr_SetString(PyExc_RuntimeError, "Building output arrays failed.");
        Py_XDECREF(np_mstar);
        Py_XDECREF(np_sfr);
        Py_XDECREF(np_passive);
        return NULL;
    }
    
    // Get references to data structures for galaxy properties
    cat.nhalos = N;
    cat.mhalo = (float*)PyArray_DATA(np_mhalo);
    cat.z = (float*)PyArray_DATA(np_z);
    cat.mstar = (float*)PyArray_DATA(np_mstar); // uninitialised
    cat.sfr = (float*)PyArray_DATA(np_sfr); // uninitialised
    cat.passive = (bool*)PyArray_DATA(np_passive); // uninitialised
    
    // Traverse ghost model to add physical properties
    realise_physical_properties(&cat, p, rng);
    
    // Clean-up references
    Py_DECREF(np_mhalo);
    Py_DECREF(np_z);
    Py_DECREF(arg_params);
    Py_DECREF(dtype_mhalo);
    Py_DECREF(dtype_z);
    gsl_rng_free(rng);
    
    // Construct tuple of arrays to be returned
    PyObject *cat_columns = Py_BuildValue("OOO", 
                                          np_mstar, 
                                          np_sfr, 
                                          np_passive);
    return cat_columns;
}


static char docstring_add_optical_mags[] = 
  "Create a realisation of optical magnitudes, with or without dust attenuation,\n"
  "for a set of input galaxies.\n\n"
  "Parameters\n"
  "----------\n"
  "Parameters\n"
  "Parameters\n";

static PyObject* add_optical_mags(PyObject *self, PyObject *args, PyObject *kwargs){
    /*
    Create a realisation of optical magnitudes, with or without dust 
    attenuation, for a set of input galaxies.
    */
    
    // Initialise RNG
    gsl_rng *rng;
    gsl_rng_env_setup();
    rng = gsl_rng_alloc( gsl_rng_default );
    
    // Galaxy catalogue input arrays
    PyObject *arg_mstar, *arg_sfr, *arg_z, *arg_params; // *arg_band;
    char *arg_band;
    int atten = 1; // Default: atten = True
    
    // Set up keyword args
    static char *kwlist[] = {"mstar", "sfr", "z", "band", "params", "atten", NULL};// "params", "atten", NULL};
    
    // Get halo catalogue arrays from input arguments
    if ( !PyArg_ParseTupleAndKeywords(args, kwargs, "OOOs|Oi", kwlist,
                                      &arg_mstar, &arg_sfr, &arg_z, 
                                      &arg_band, &arg_params, &atten))
    {
        PyErr_SetString(PyExc_RuntimeError, "Failed to interpret input arguments.");
        return NULL;
    }
    
    // Parse the 'band' argument
    char band = '_';
    for (int i=0; i < NUM_BANDS; i++){
        if(*arg_band == BAND_NAMES[i]){
            band = BAND_NAMES[i];
        }
    }
    // Test in case band wasn't found
    if (band == '_'){
        PyErr_SetString(PyExc_RuntimeError, "Specified band not recognised.");
        return NULL;
    }
    
    // Populate parameter struct with default parameter values
    struct Params p;
    default_params(&p);
    
    // Set parameters defined in input parameter dict
    set_params(arg_params, &p);
    
    // Convert input arguments to numpy arrays
    PyArrayObject *np_mstar = (PyArrayObject*)PyArray_FROM_O(arg_mstar);
    PyArrayObject *np_sfr = (PyArrayObject*)PyArray_FROM_O(arg_sfr);
    PyArrayObject *np_z = (PyArrayObject*)PyArray_FROM_O(arg_z);
    
    // Test input data type; must be NPY_FLOAT32
    PyArray_Descr *dtype_mstar = PyArray_DTYPE(np_mstar);
    PyArray_Descr *dtype_sfr = PyArray_DTYPE(np_sfr);
    PyArray_Descr *dtype_z = PyArray_DTYPE(np_z);
    
    if (dtype_mstar->type_num != NPY_FLOAT32){
        PyErr_SetString( PyExc_RuntimeError, 
                         "Data type of input 'mstar' not allowed. "
                         "Must be np.float32.");
        return NULL;
    }
    if (dtype_sfr->type_num != NPY_FLOAT32){
        PyErr_SetString( PyExc_RuntimeError, 
                         "Data type of input 'sfr' not allowed. "
                         "Must be np.float32.");
        return NULL;
    }
    if (dtype_z->type_num != NPY_FLOAT32){
        PyErr_SetString( PyExc_RuntimeError, 
                         "Data type of input 'z' not allowed. "
                         "Must be np.float32.");
        return NULL;
    }
    int nhalos = (int)PyArray_DIM(np_mstar, 0); // Get length of input arrays
    
    // Create new intrinsic opt. mag. ndarray and provide data access pointer
    int ndim = 1;
    npy_intp shape[1] = {nhalos};
    PyObject *np_mag_int = PyArray_SimpleNew(ndim, shape, NPY_FLOAT);
    if (np_mag_int == NULL){
        PyErr_SetString(PyExc_RuntimeError, "Building output arrays failed.");
        Py_XDECREF(np_mag_int);
        return NULL;
    }
    
    // Also setup ndarray for dust-attenuated magnitudes, if requested
    PyObject *np_mag_atten = Py_None;
    if (atten){
        np_mag_atten = PyArray_SimpleNew(ndim, shape, NPY_FLOAT);
        if (np_mag_atten == NULL){
            PyErr_SetString(PyExc_RuntimeError, "Building output arrays failed.");
            Py_XDECREF(np_mag_atten);
            return NULL;
        }
    }
    
    // Get references to data structures for galaxy properties
    float *mstar = (float*)PyArray_DATA(np_mstar);
    float *sfr = (float*)PyArray_DATA(np_sfr);
    float *z = (float*)PyArray_DATA(np_z);
    float *mag_int = (float*)PyArray_DATA(np_mag_int); // uninitialised
    float *mag_atten;
    if (atten){
      mag_atten = (float*)PyArray_DATA(np_mag_atten); // uninitialised
    }
    // Draw intrinsic optical magnitudes
    #pragma omp parallel for
    for(int i=0; i < nhalos; i++){
        mag_int[i] = draw_optical_mag_intrinsic( sfr[i], mstar[i], band, 
                                                 z[i], p, rng);
    }
    
    // Draw attenuated magnitudes, if requested
    if (atten){
        #pragma omp parallel for
        for(int i=0; i < nhalos; i++){
            mag_atten[i] = draw_optical_mag_atten( mag_int[i], mstar[i], band, 
                                                   z[i], p, rng );
        } // end loop over galaxies
    }
    
    // Clean-up references
    Py_DECREF(np_mstar);
    Py_DECREF(np_sfr);
    Py_DECREF(np_z);
    Py_DECREF(arg_params);
    Py_DECREF(dtype_mstar);
    Py_DECREF(dtype_sfr);
    Py_DECREF(dtype_z);
    gsl_rng_free(rng);
    
    // Construct tuple of arrays to be returned
    PyObject *mags = Py_None;
    if (atten){
        mags = Py_BuildValue("OO", np_mag_int, np_mag_atten);
    }else{
        mags = Py_BuildValue("O", np_mag_int);
    }
    return mags;
}


////////////////////////////////////////////////////////////////////////////////
// Define public methods and initialisation routine
////////////////////////////////////////////////////////////////////////////////

static struct PyMethodDef methods[] = {
    {"add_physical_properties", add_physical_properties, METH_VARARGS, 
        docstring_add_physical_properties},
    {"add_optical_mags", (PyCFunction)add_optical_mags, METH_VARARGS|METH_KEYWORDS, 
        docstring_add_optical_mags},
    {NULL, NULL, 0, NULL} // Sentinel block
};

PyMODINIT_FUNC initghost(void){
    (void)Py_InitModule("ghost", methods);
    import_array();
}
