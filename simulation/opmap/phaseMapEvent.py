import numpy as np
import scipy
import scipy.interpolate as interpolate
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import hilbert
from numba.decorators import autojit

from .videoData import VideoData
from .f_peakdetect import peakdetect
from .f_pixel import f_pixel_diff_thre
from .phaseMap import PhaseMap
from .util import phaseComplement

class PhaseMapEvent(PhaseMap):
    
    def __init__(self, vmem, width, f_event = f_pixel_diff_thre, **kwargs):
        
        super(PhaseMapEvent, self).__init__(vmem, width)
        
        v = vmem.data[:,::self.shrink,::self.shrink]
        
        def f_pixelwise(v):
            L,H,W = v.shape
            for i in range(H):
                for j in range(W):
                    ts = v[:,i,j]
                    t_event = f_event(ts, kwargs)
                    for n in range(1,len(t_event)):
                        t_start = t_event[n-1]
                        t_end = t_event[n]
                        dpdt = 2 * np.pi / float(t_end - t_start)
                        ts_p = dpdt * np.arange(L-t_start).astype(np.float64) - np.pi
                        self.data[t_start:, i, j] = ts_p
        autojit(f_pixelwise)(v)
        self.data = phaseComplement(self.data)
        
        if 'afterLast' in kwargs.keys():
            if kwargs['afterLast'] is 'continue':
                while( np.sum(( self.data > 2*np.pi )*1) + np.sum(( self.data < -2*np.pi )*1) > 0 ):
                    self.data = phaseComplement(self.data)


if __name__ == '__main__':
    v = VideoData(100,200,200)
    v.data[[10,30,50,70,90]] = 100
    pgrad = PhaseMapEvent(v, width=2, f_event=f_pixel_diff_thre, diff_thre=10, inter_min = 15)
    plt.plot(pgrad.data[:,0,0])