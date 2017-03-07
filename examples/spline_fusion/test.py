import os
import numpy
from cost_functions.wrappers import (py_spline_fusion, py_spline_eval)

from IPython import embed

numpy.random.seed(123)

#print os.getpid()
#embed()

class Spline(object):
    _C = numpy.array([
        [6, 0, 0, 0],
        [5, 3,-3, 1],
        [1, 3, 3,-2],
        [0, 0, 0, 1],
    ], dtype='f8')/6.

    def __init__(self, ts_offset, ts_step, params):
        self.ts_offset = ts_offset
        self.ts_step = ts_step
        self._params = params

    def weights(self, local_ts):
        ts2 = local_ts*local_ts 
        return numpy.c_[numpy.ones_like(local_ts), local_ts, ts2, ts2*local_ts]
    
    def trans(self, xyz, ts, return_se3=False): 
        xyz = xyz.reshape(-1,3)
        _ts = ts - self.ts_offset    # local ts
        u = self.weights(_ts)        # nx4
        u = u.dot(self._C.T)         # nx4

        idxs = (_ts/self.ts_step).astype('i4') # nx1
        assert idxs.max()<self._params.shape[0]-4
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

    params = numpy.empty((15,7), dtype='f8')
    return {
        'ref':      ref,
        'ref_ts':   ref_ts,
        'src':      src,
        'src_norm': src_norm,
        'src_ts':   src_ts,
        'tiles':    tiles,
        'params':   params,
        'ts_offset':    ts_offset,
        'ts_step':      ts_step,
    }




if __name__=='__main__':
    if 1:
        _data = _prepare_data()
    
    if 0:
        numpy.savez('d:/workspace/spline_fusion.npz',**_data)
    
    if 1:   #-- fusion
        _args = [ _data[k] for k in 'ref,ref_ts,src,src_norm,src_ts,tiles,params'.split(',') ]
        py_spline_fusion(*_args)
        
        #print params
        
    if 1:   #-- apply spline
        spline = Spline(_data['ts_offset'], _data['ts_step'], _data['params'])
        ref_h = spline.trans(_data['ref'], _data['ref_ts'])
        ref_h2, ref_se3 = spline.trans(_data['ref'], _data['ref_ts'], return_se3=True)
        
        assert numpy.allclose(ref_h, ref_h2)

    embed()

