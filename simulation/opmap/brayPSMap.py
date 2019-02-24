import numpy as np
import scipy
from scipy.signal import convolve2d

from videoData import VideoData

class BrayPSMap( VideoData ):

    def __init__(self, phasemap, thre):
        assert thre > 0 and thre < 1.0

        super(BrayPSMap, self).__init__(*phasemap.data.shape)
        
        nablaX = np.array([ [-0.5,  0, 0.5], [ -1, 0, 1], [-0.5,  0, 0.5] ], dtype = np.float32)
        nablaY = np.array([ [ 0.5,  1, 0.5], [  0, 0, 0], [-0.5, -1,-0.5] ], dtype = np.float32)
        
        def phaseComplement(phase_diff):
            out = np.copy(phase_diff)
            mask = (phase_diff > np.pi )*1
            out -= mask*(2*np.pi)
            mask = (phase_diff < -np.pi )*1
            out += mask*(2*np.pi)
            return out
        
        diffX = np.zeros_like(phasemap.data)
        diffY = np.zeros_like(phasemap.data)
        diffX[:,:,1:] = phaseComplement( phasemap.data[:,:,1:] - phasemap.data[:,:,0:-1])
        diffY[:,1:,:] = phaseComplement( phasemap.data[:,1:,:] - phasemap.data[:,0:-1,:])
        
        for frame in range(self.data.shape[0]):
            img_dx = diffX[frame, :,:]
            img_dy = diffY[frame, :,:]
            img_x = convolve2d(img_dx, nablaY, boundary='symm', mode='same')
            img_y = convolve2d(img_dy, nablaX, boundary='symm', mode='same')
            self.data[frame, :, :] = ((np.abs(img_y+img_x)/(2*np.pi))>thre)*1.0

        self.roi = np.copy(phasemap.roi)
        self.roi = scipy.ndimage.binary_erosion(self.roi, structure=np.ones((2,2))).astype(phasemap.roi.dtype)
        self.data *= self.roi

        self.vmin = 0.0
        self.vmax = 1.0
        self.cmap = 'gray'
