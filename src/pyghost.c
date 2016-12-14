
//define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include "Python.h"
#include "numpy/arrayobject.h"
#include "ghmock.h"

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
    //----------------------------------------
    
    // Raise warning if error occurred
    if(PyErr_Occurred()){
        PyErr_SetString(PyExc_ValueError, 
            "set_params() failed: a value in 'params' probably couldn't be cast to the right type.");
        PyErr_Print();
        return;
    }
}

static PyObject* realise(PyObject *dummy, PyObject *args){
    /*
    Create a realisation of galaxy properties on top of an input halo catalogue.
    */
    
    // Initialise RNG
    gsl_rng *rng;
    gsl_rng_env_setup();
    rng = gsl_rng_alloc( gsl_rng_default );
    
    // Halo catalogue input arrays
    PyObject *arg_mhalo, *arg_z, *arg_params;
    
    // Get halo catalogue arrays from input arguments
    if (!PyArg_ParseTuple(args, "OOO", &arg_mhalo, &arg_z, &arg_params)){
        PyErr_SetString(PyExc_RuntimeError, "Failed to interpret input arguments.");
        return NULL;
    }
    
    // Populate parameter struct with default parameter values
    struct Params p;
    default_params(&p);
    
    // Set parameters defined in input parameter dict
    set_params(arg_params, &p);
    
    // Construct Catalogue structure to manage variables
    struct Catalogue cat;
    
    // Convert input arguments to numpy arrays and expose C pointers to data
    PyObject *np_mhalo = PyArray_FROM_OTF(arg_mhalo, NPY_DOUBLE, NPY_IN_ARRAY);
    PyObject *np_z = PyArray_FROM_OTF(arg_z, NPY_DOUBLE, NPY_IN_ARRAY);
    int N = (int)PyArray_DIM(np_mhalo, 0); // Get length of input arrays
    
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
