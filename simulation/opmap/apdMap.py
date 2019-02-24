# -*- coding: utf-8 -*-

import numpy as np
from .videoData import VideoData
from .f_pixel import f_pixel_apd
from scipy.ndimage.filters import gaussian_filter1d

class APDMap( VideoData ):

    def __init__(self, vmem, width = 64, max_beats=10, thre_type='peak', thre_up=0.8, thre_down=0.2, sigma_t = 5):
        
        shrink = int(vmem.data.shape[2] / width)
        self.shrink = shrink
        
        size_org = vmem.data.shape
        super(APDMap, self).__init__(max_beats,size_org[1]//shrink, size_org[2]//shrink)

        V = vmem.data[:,::shrink,::shrink]
        if sigma_t > 1:
            V = np.apply_along_axis(gaussian_filter1d, 0, V, sigma=sigma_t)
            
        self.data = (np.apply_along_axis(
                f_pixel_apd, 0, V, 
                max_beats = max_beats,
                thre_type=thre_type, 
                thre_up=thre_up, 
                thre_down=thre_down)).astype(self.data.dtype)
        
        self.roi = np.array(vmem.roi[::shrink, ::shrink])
        self.data *= self.roi

        self.vmin = 0
        self.vmax = self.data.max()
        self.cmap = 'binary'
        return
