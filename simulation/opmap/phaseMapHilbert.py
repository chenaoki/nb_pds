import numpy as np
import scipy
import scipy.interpolate as interpolate
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import hilbert
from .videoData import VideoData
from .f_peakdetect import peakdetect
from .f_pixel import f_pixel_mean, f_pixel_phase

from .phaseMap import PhaseMap

class PhaseMapHilbert( PhaseMap ):

    def __init__(self, vmem, width = 128, sigma_xy = 32, sigma_t = 5):
        
        super(PhaseMapHilbert, self).__init__(vmem, width)
                
        V = vmem.data[:,::self.shrink,::self.shrink]
        Vmean = np.apply_along_axis(f_pixel_mean, 0, V)
        
        # spatial filtering of Vmean
        if sigma_xy > 1:
            for frame in range(len(Vmean)):
                Vmean[frame,:,:] = gaussian_filter(Vmean[frame,:,:], sigma = sigma_xy)
                
        Vamp = V-Vmean
        
        # temporal filtering
        if sigma_t > 1:
            Vamp = np.apply_along_axis(gaussian_filter1d, 0, Vamp, sigma = sigma_t)
        self.data = np.apply_along_axis(f_pixel_phase, 0, arr=Vamp)
        self.data *= self.roi

        self.Vmean = Vmean
        return
