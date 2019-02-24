import numpy as np
import scipy
import scipy.interpolate as interpolate
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import hilbert
from .videoData import VideoData
from .util import phaseComplement

class PhaseMap( VideoData ):

    def __init__(self, vmem, width = 128):
        
        self.shrink = int(vmem.data.shape[2] / width)
        size_org = vmem.data.shape
        super(PhaseMap, self).__init__(size_org[0],size_org[1]//self.shrink, size_org[2]//self.shrink)
        self.roi = np.array(vmem.roi[::self.shrink, ::self.shrink])
        self.vmin = -np.pi
        self.vmax = np.pi
        self.cmap = 'jet_r'
        return
    
    def smooth(self, size = 5):
        for frame in range( self.data.shape[0]):
            img_cos = np.cos(self.data[frame, :,:]) 
            img_sin = np.sin(self.data[frame, :,:]) 
            img_cos = scipy.ndimage.filters.uniform_filter(img_cos, size=size, mode='constant')
            img_sin = scipy.ndimage.filters.uniform_filter(img_sin, size=size, mode='constant')
            self.data[frame,:,:] = np.angle(img_cos+1j*img_sin)

    def smooth_median(self, size = 4):
        assert size > 0
        new_data = np.zeros_like(self.data)
        for frame in range( self.data.shape[0]):
            if frame % 10 == 0 : print(frame)
            for n in range(self.data.shape[1])[size:-size]:
                for m in range(self.data.shape[2])[size:-size]:
                    base = self.data[frame, n, m]
                    target = self.data[frame, n-size:n+size+1, m-size:m+size+1]
                    difference = phaseComplement(target-base)
                    diff = scipy.signal.medfilt2d(difference, kernel_size=size*2+1)[size,size]
                    new_data[frame, n, m] = phaseComplement( base +  diff)
                    #new_data[frame, n, m] = phaseComplement( base + np.mean(difference.flatten()) )
        self.data = new_data

