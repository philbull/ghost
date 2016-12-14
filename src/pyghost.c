
//define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "numpy/arrayobject.h"
#include "ghmock.h"

/*
IN PROGRESS
void extract_params(PyObject *params, struct Params *p){
    / *
    Convert a Python parameter dictionary to a C struct.
    * /
    PyObject*
    
    int PyDict_Contains();
    
    PyObject* PyDict_GetItem(PyObject *p, PyObject *key)
    
    p->ms_cen_logM1;
    
    ms_cen_logM1;
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
    * /
}
*/

static PyObject* realise(PyObject *dummy, PyObject *args){
    /*
    Create a realisation of galaxy properties on top of an input halo catalogue.
    */
    
    // Initialise RNG
    gsl_rng *rng;
    gsl_rng_env_setup();
    rng = gsl_rng_alloc( gsl_rng_default );
    
    // Get default model parameters (FIXME)
    struct Params p;
    default_params(&p);
    
    // Halo catalogue input arrays
    PyObject *arg_mhalo, *arg_z;
    
    // Get halo catalogue arrays from input arguments
    if (!PyArg_ParseTuple(args, "OO", &arg_mhalo, &arg_z)){
        PyErr_SetString(PyExc_RuntimeError, "Failed to interpret input arguments.");
        return NULL;
    }
    
    // Construct Catalogue structure to manage variables
    struct Catalogue cat;
    
    // Convert input arguments to numpy arrays and expose C pointers to data
    PyObject *np_mhalo = PyArray_FROM_OTF(arg_mhalo, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *np_z = PyArray_FROM_OTF(arg_z, NPY_DOUBLE, NPY_IN_ARRAY);
    // Get length of input arrays
    int N = (int)PyArray_DIM(np_mhalo, 0);
    
    // Create new ndarrays and provide data access pointers for C code
    int ndim = 1;
    npy_intp shape[1] = {N};
    PyObject *np_mstar = PyArray_SimpleNew(ndim, shape, NPY_DOUBLE);
    PyObject *np_sfr = PyArray_SimpleNew(ndim, shape, NPY_DOUBLE);
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
    cat.mhalo = (double*)PyArray_DATA(np_mhalo);
    cat.z = (double*)PyArray_DATA(np_z);
    cat.mstar = (double*)PyArray_DATA(np_mstar); // uninitialised
    cat.sfr = (double*)PyArray_DATA(np_sfr); // uninitialised
    cat.passive = (bool*)PyArray_DATA(np_passive); // uninitialised
    
    // Traverse ghost model
    realise_catalogue(&cat, p, rng);
    
    // Construct tuple of arrays, to be returned
    PyObject *cat_columns = Py_BuildValue("OOO", 
                                          np_mstar, 
                                          np_sfr, 
                                          np_passive);
    return cat_columns;
}

static struct PyMethodDef methods[] = {
    {"realise", realise, METH_VARARGS, "Make a realisation of a halo catalogue."},
    {NULL, NULL, 0, NULL} // Sentinel block
};

PyMODINIT_FUNC initghost(void){
    (void)Py_InitModule("ghost", methods);
    import_array();
}
