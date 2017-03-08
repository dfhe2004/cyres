import cython
from libcpp cimport bool
#from libcpp.vector cimport vector
#from cpython.object cimport PyObject

cimport numpy as np

#from IPython import embed
#import sys

import numpy as np
np.import_array()
  


#cdef extern from "python.h":
#    void Py_IncRef(PyObject*)
#    void Py_DecRef(PyObject*)

cdef extern from "spline_fusion.h":
    void spline_fusion(const double* ref, const double* ref_ts, const double* src, const double* src_norm, const double* src_ts, const int* tiles, size_t num_tiles, double* params_data, size_t num_params, size_t max_solver_time)
    
cdef extern from "se3_spline.h":
    void spline_evaluate(double* out, double* se3, const double* xyz, const double* weights,  const double* knots)
    void se3_to_matrix(double* out, double* se3)

#--- interface 
cpdef py_spline_fusion(np.ndarray ref, np.ndarray ref_ts
    , np.ndarray src, np.ndarray src_norm, np.ndarray src_ts, np.ndarray tiles
    , np.ndarray params, size_t max_solver_time=60): 
    
    #-- check size
    _sz = ref.shape[0]
    
    assert ref.shape[1] == 3            # nx3
    assert ref_ts.shape[0] == _sz       # nx1
    
    assert src.shape[0]==_sz and src.shape[1]==3            # nx3
    assert src_norm.shape[0]==_sz and src_norm.shape[1]==3  # nx3
    assert src_ts.shape[0]==_sz         # nx1
       
    assert tiles.shape[1]==2            # tx2
    assert params.shape[1]==7, 'params.shape %s|%s'%(params.shape[0],params.shape[1])           # mx7


    #-- prepare data
    cdef np.ndarray _tmp_ref = np.ascontiguousarray(ref, dtype=np.double)
    cdef const double* _ref = <const double*> _tmp_ref.data

    cdef np.ndarray _tmp_ref_ts = np.ascontiguousarray(ref_ts, dtype=np.double)
    cdef const double* _ref_ts = <const double*> _tmp_ref_ts.data

    cdef np.ndarray _tmp_src = np.ascontiguousarray(src, dtype=np.double)
    cdef const double* _src = <const double*> _tmp_src.data

    cdef np.ndarray _tmp_src_norm = np.ascontiguousarray(src_norm, dtype=np.double)
    cdef const double* _src_norm = <const double*> _tmp_src_norm.data

    cdef np.ndarray _tmp_src_ts = np.ascontiguousarray(src_ts, dtype=np.double)
    cdef const double* _src_ts = <const double*> _tmp_src_ts.data

    cdef np.ndarray _tmp_tiles = np.ascontiguousarray(tiles, dtype=np.int32)
    cdef const int* _tiles = <const int*> _tmp_tiles.data

    cdef np.ndarray _tmp_params = np.ascontiguousarray(params, dtype=np.double)
    cdef double* _params = <double*> _tmp_params.data

    cdef size_t num_tiles = tiles.shape[0]
    cdef size_t num_params = params.shape[0]

    #-- pass to spline_fusion
    spline_fusion(
        _ref,  _ref_ts, 
        _src, _src_norm, _src_ts, 
        _tiles, num_tiles, 
        _params, num_params, max_solver_time
    )


cpdef py_spline_eval(np.ndarray xyz, np.ndarray weights, np.ndarray knots, bool return_se3=False):
    cdef np.ndarray _tmp_out = np.empty_like(xyz, dtype=np.double)
    cdef double* _out = <double*> _tmp_out.data
    
    cdef np.ndarray _tmp_xyz = np.ascontiguousarray(xyz, dtype=np.double)
    cdef const double* _xyz = <const double*> _tmp_xyz.data

    cdef np.ndarray _tmp_weights = np.ascontiguousarray(weights, dtype=np.double)
    cdef const double* _weights = <const double*> _tmp_weights.data
    
    cdef np.ndarray _tmp_knots = np.ascontiguousarray(knots, dtype=np.double)
    cdef const double* _knots = <const double*> _tmp_knots.data
    
    
    if(not return_se3):
        spline_evaluate(_out, NULL, _xyz, _weights, _knots)
        return _tmp_out
    
    cdef np.ndarray _tmp_se3 = np.empty(7, dtype=np.double)
    cdef double* _se3 = <double*> _tmp_se3.data
    spline_evaluate(_out, _se3, _xyz, _weights, _knots)
    return _tmp_out, _tmp_se3
    
cpdef py_se3_matrix(np.ndarray se3):
    cdef n = se3.shape[0]
    cdef np.ndarray m = np.empty_like((n,4,4), dtype=np.double)
    cdef double* _m = <double*> m.data
    cdef double* _src = <double*>se3.data

    for i in xrange(n):
        se3_to_matrix(_m, _src)    
        _m += 16
        _src += 7
    
    return m
