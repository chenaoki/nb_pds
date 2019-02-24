import numpy as np
from scipy import ndimage
from .cmap import bipolar
from .videoData import VideoData

class VmemMap( VideoData ):

    def __init__(self, rawcam):
        shape = rawcam.data.shape
        super(VmemMap, self).__init__(shape[0], shape[1], shape[2])

        im_max = np.max(rawcam.data, axis=0)
        im_min = np.min(rawcam.data, axis=0)
        self.im_range = (im_max - im_min) + (im_max == im_min) * 1
        self.data_org = 2.0 * (im_max - rawcam.data ) / self.im_range - 1.0
        self.diff_max = rawcam.vmax - rawcam.vmin
        self.roi_org = np.copy(rawcam.roi)
        self.roi = np.copy(rawcam.roi)
        self.data = self.data_org*self.roi

        self.vmin = -1.0
        self.vmax = 1.0
        self.cmap = bipolar(neutral=0, lutsize=1024)
        return

    def setDiffRange(self, diff_min=None, diff_max=None):
        self.roi = np.copy(self.roi_org) # reset
        if diff_min is None :
          diff_min = 0
        if diff_max is None :
          diff_max = self.diff_max
        self.roi *= (self.im_range>=diff_min)*1
        self.roi *= (self.im_range<=diff_max)*1
        self.data = self.data_org*self.roi
        return

    def smooth(self, size):
        assert size > 0
        for frame in range( self.data.shape[0]):
            self.data[frame,:,:] = ndimage.gaussian_filter(self.data[frame,:,:], sigma = size)
        return
