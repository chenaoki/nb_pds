import numpy as np
import scipy
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import hilbert
from .videoData import VideoData
from .f_peakdetect import peakdetect
from .f_pixel import f_pixel_mean, f_pixel_phase

from .phaseMap import PhaseMap

class PhaseMapFTDT( PhaseMap ):

    def __init__(self, vmem, v_mean, dt, width = 128, sigma_t = 1):
        
        super(PhaseMapFTDT, self).__init__(vmem, width)
        
        self.v_mean  = v_mean 
        self.dt = dt
                
        V = vmem.data[:,::self.shrink,::self.shrink]
        # temporal filtering
        if sigma_t > 1:
            V = np.apply_along_axis(gaussian_filter1d, 0, V, sigma = sigma_t)
        self.V = V
        
        self.calc_phase()
        return
    
    def calc_phase(self):
        V_x = np.zeros_like(self.V)
        V_y = np.zeros_like(self.V)
        V_x[:self.dt,:,:] = 1
        V_x[:self.dt,:,:] = 0
        V_x[self.dt:,:,:] = self.V[:-self.dt,:,:] - self.v_mean
        V_y[self.dt:,:,:] = self.V[ self.dt:,:,:] - self.v_mean
        V_comp = 1j * V_x + V_y
        self.data = np.angle(V_comp)
        self.data *= self.roi
        
    def set_param(self, vmean=None, dt=None):
        if vmean is not None:
            self.vmean = vmean
        if dt is not None:
            self.dt = dt
        self.calc_phase()
        
    def draw_phase_portrait(self, pos_x=None, pos_y=None):
        L,H,W = self.V.shape
        if pos_x is None: pos_x = W//2
        if pos_y is None: pos_y = H//2
        vmin = self.V[:,pos_y, pos_x].min()
        vmax = self.V[:,pos_y, pos_x].max()
        
        V_x = np.zeros_like(self.V)
        V_y = np.zeros_like(self.V)
        V_x[:self.dt,:,:] = vmin
        V_y[:self.dt,:,:] = vmin
        V_x[self.dt:,:,:] = self.V[:-self.dt,:,:]
        V_y[self.dt:,:,:] = self.V[ self.dt:,:,:]
        
        plt.axis('equal')
        plt.plot( V_x[:,pos_y, pos_x], V_y[:,pos_y, pos_x], c='red')
        #plt.plot( (self.v_mean, self.v_mean), (vmin, vmax), c='k', linestyle='dashed', linewidth=0.5)
        #plt.plot( (vmin, vmax), (self.v_mean, self.v_mean), c='k', linestyle='dashed', linewidth=0.5)
        plt.plot( (vmin, vmax), (vmin, vmax), c='k', linestyle='dashed', linewidth=0.5)
        plt.plot( (vmin, vmin), (vmin, vmax), c='k', linestyle='dashed', linewidth=0.5)
        plt.plot( (vmax, vmax), (vmin, vmax), c='k', linestyle='dashed', linewidth=0.5)
        plt.plot( (vmin, vmax), (vmin, vmin), c='k', linestyle='dashed', linewidth=0.5)
        plt.plot( (vmin, vmax), (vmax, vmax), c='k', linestyle='dashed', linewidth=0.5)
        plt.scatter((self.v_mean),(self.v_mean),c='k',s=100)
        plt.title('phase portrait')
        #plt.axis( [vmin, vmax,vmin, vmax] )
        plt.show()

        

        
