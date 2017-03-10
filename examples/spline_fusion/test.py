import os
import numpy
from cost_functions.wrappers import (py_spline_fusion, py_spline_eval)

from IPython import embed

numpy.random.seed(123)

print os.getpid()
embed()

class Spline(object):
    _C = numpy.array([
        [6, 0, 0, 0],
        [5, 3,-3, 1],
        [1, 3, 3,-2],
        [0, 0, 0, 1],
    ], dtype='f8')/6.

    def __init__(self, params):
        self._params = params

    def weights(self, u):
        u2 = u*u 
        return numpy.c_[numpy.ones_like(u), u, u2, u*u2]
    
    def trans(self, xyz, local_ts, return_se3=False): 
        xyz = xyz.reshape(-1,3)
        u = local_ts-numpy.floor(local_ts)
        u = self.weights(u)          # nx4
        u = u.dot(self._C.T)                # nx4

        idxs = (local_ts).astype('i4') # nx1
        assert idxs.max()<self._params.shape[0]-3
        rs = []
        for i, idx in enumerate(idxs):
            val = py_spline_eval(xyz[i], u[i], self._params[idx:idx+4], return_se3)
            rs.append(val)    
        
        if not return_se3:
            return numpy.vstack(rs)

        xyz = numpy.vstack([e[0] for e in rs])
        se3 = numpy.vstack([e[1] for e in rs])
        return xyz, se3




def _prepare_data():
    _ts = numpy.random.randn(30)
    _ts.sort()
    _ts -= _ts[0]/_ts[-1]*10        # 10s
    
    ts_offset = _ts[0]
    ts_step = 1.0
    _ts = (_ts-ts_offset)/ts_step   # convert to local ts

    ref = numpy.random.randn(10,3)
    ref_ts = _ts[4:14]

    src = numpy.random.randn(10,3)
    src_norm = numpy.random.randn(10,3)
    src_norm = src_norm/numpy.linalg.norm(src_norm,axis=1).reshape(-1,1)
    src_ts = _ts[14:24]

    tiles = numpy.array([
        [0,3],
        [3,10],
    ], dtype='i4')

    #params = numpy.empty((15,7), dtype='f8')
    params = numpy.empty((8,7), dtype='f8')
    ee = numpy.r_[0,0,0,1,0,0,0]
    params[:] = ee

    
    return {
        'ref':      ref,
        'ref_ts':   ref_ts,
        'src':      src,
        'src_norm': src_norm,
        'src_ts':   src_ts,
        'tiles':    tiles,
        'params':   params,
        #'ts_offset':    ts_offset,
        #'ts_step':      ts_step,
        'num_params':   params.shape[0],
        'diff':         ((ref - src)*src_norm).sum(1),

    }




if __name__=='__main__':
    if 1:
        _data = _prepare_data()
    
    if 1:
        numpy.savez('d:/workspace/spline_fusion.npz',**_data)
    
    if 1:   #-- fusion
        _args = [ _data[k] for k in 'ref,ref_ts,src,src_norm,src_ts,tiles,params'.split(',') ]
        py_spline_fusion(*_args)
        
        #print params
        
    if 0:   #-- apply spline
        spline = Spline(_data['ts_offset'], _data['ts_step'], _data['params'])
        ref_h = spline.trans(_data['ref'], _data['ref_ts'])
        ref_h2, ref_se3 = spline.trans(_data['ref'], _data['ref_ts'], return_se3=True)
        
        assert numpy.allclose(ref_h, ref_h2)

    embed()

