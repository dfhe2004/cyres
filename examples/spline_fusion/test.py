import os

import numpy
from cost_functions.wrappers import py_spline_fusion

from IPython import embed

numpy.random.seed(123)

#cpdef py_spline_fusion(np.ndarray ref, np.ndarray ref_norm, np.ndarray ref_ts
#    , np.ndarray src, np.ndarray src_ts, np.ndarray tiles
#    , np.ndarray params, double ts_offset, double ts_step)

_ts = numpy.random.randn(30)
_ts.sort()

_ts -= _ts[0]/_ts[-1]*10        # 10s

ref = numpy.random.randn(10,3)
ref_norm = numpy.random.randn(10,3)
ref_norm = ref_norm/numpy.linalg.norm(ref_norm,axis=1).reshape(-1,1)
ref_ts = _ts[4:14]

src = numpy.random.randn(10,3)
src_ts = _ts[14:24]

tiles = numpy.array([
    [0,3],
    [3,10],
], dtype='i4')

ts_offset = _ts[0]
ts_step = 1.0
params = numpy.empty((15,7), dtype='f8')

numpy.savez('d:/workspace/spline_fusion.npz',**{
    'ref':  ref,
    'ref_norm': ref_norm,
    'ref_ts':   ref_ts,
    'src':  src,
    'src_ts':   src_ts,
    'tiles':    tiles,
    'params':   params,
    'ts_offset':    ts_offset,
    'ts_step':      ts_step,

})

print os.getpid()
embed()

py_spline_fusion(ref, ref_norm, ref_ts
    , src, src_ts, tiles
    , params, ts_offset, ts_step    
)
print params
embed()

