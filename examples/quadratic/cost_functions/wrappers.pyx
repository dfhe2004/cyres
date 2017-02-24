
from cyres cimport CostFunction  #, LossFunction
cimport ceres
cimport numpy as np
#from cython cimport view

from IPython import embed
#from libc.stdlib cimport malloc
#import sys

np.import_array()


cdef extern from "cost_functions.h":
    ctypedef void (*Method_1x1f8)(object pyfun, void* x0, void* residual, object args)
    ctypedef void (*Method_2x1f8)(object pyfun, void* x0, void* x1, void* residual, object args)
    ctypedef void (*Method_dyn_num_f8)(object pyfun, void* x0, void* residual, object args, int, int)
    

    ceres.CostFunction* createCostFunAutoDiff_1x1f8(Method_1x1f8 cyfun, object pyfun, object args)
    ceres.CostFunction* createCostFunAutoDiff_2x1f8(Method_2x1f8 cyfun, object pyfun, object args)

    ceres.CostFunction* createNumericDiffCostFun_1x1f8(Method_1x1f8 cyfun, object pyfun, object args)
    ceres.CostFunction* createNumericDiffCostFun_2x1f8(Method_2x1f8 cyfun, object pyfun, object args)
    ceres.CostFunction* createDynamicNumericDiffCostFun(Method_dyn_num_f8 cyfun, object pyfun, object args, int, int)



#-- call back 
cdef void callback_1x1f8(object pyfun, void* x0, void* rs, object args):
    cdef np.npy_intp _shape[1]
    
    _shape[0] = <np.npy_intp>1 
    arr_x0 = np.PyArray_SimpleNewFromData(1, _shape, np.NPY_FLOAT64, x0)
    arr_rs = np.PyArray_SimpleNewFromData(1, _shape, np.NPY_FLOAT64, rs)
    pyfun(arr_x0, arr_rs, args)


cdef void callback_2x1f8(object pyfun, void* x0, void* x1, void* rs, object args):
    cdef np.npy_intp _shape[1]
    
    _shape[0] = <np.npy_intp>1 
    arr_x0 = np.PyArray_SimpleNewFromData(1, _shape, np.NPY_FLOAT64, x0)
    arr_x1 = np.PyArray_SimpleNewFromData(1, _shape, np.NPY_FLOAT64, x1)
    arr_rs = np.PyArray_SimpleNewFromData(1, _shape, np.NPY_FLOAT64, rs)
    pyfun(arr_x0, arr_x1, arr_rs, args)


cdef void callback_dyna_num_f8(object pyfun, void* x0, void* rs, object args, int nParams, int nRes):
    cdef np.npy_intp _shape[1]
    
    _shape[0] = <np.npy_intp>nParams 
    cdef double** ptr = <double**>x0
    arr_x0 = np.PyArray_SimpleNewFromData(1, _shape, np.NPY_FLOAT64, ptr[0])

    _shape[0] = <np.npy_intp>nRes 
    arr_rs = np.PyArray_SimpleNewFromData(1, _shape, np.NPY_FLOAT64, rs)
    pyfun(arr_x0, arr_rs, args)




#--- interface 
cdef class SimpleCostF_1x1f8(CostFunction):
    def __cinit__(self, pyfun, args=None, diff_type='numeric'):
        if diff_type=='auto':
            self._cost_function = createCostFunAutoDiff_1x1f8(callback_1x1f8, pyfun, args)
            return
        
        if diff_type=='numeric':
            self._cost_function = createNumericDiffCostFun_1x1f8(callback_1x1f8, pyfun, args)
            return
        
        

cdef class SimpleCostF_2x1f8(CostFunction):
    def __cinit__(self, pyfun, args=None, diff_type='numeric'):
        if diff_type=='auto':
            self._cost_function = createCostFunAutoDiff_2x1f8(callback_2x1f8, pyfun, args)
            return

        if diff_type=='numeric':
            self._cost_function = createNumericDiffCostFun_2x1f8(callback_2x1f8, pyfun, args)
            return



cdef class DynNumDiffCostF(CostFunction):
    def __cinit__(self, pyfun, nParams, nRes, args=None, diff_type='numeric'):
        if diff_type=='numeric':
            self._cost_function = createDynamicNumericDiffCostFun(
                callback_dyna_num_f8, pyfun, args, nParams, nRes)
            return



