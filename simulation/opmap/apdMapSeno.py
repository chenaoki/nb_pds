import numpy as np
from scipy.ndimage import gaussian_filter1d, filters
from scipy.interpolate import interp1d
from .videoData import VideoData

class APDMap( VideoData ):

    def __init__(self, vmem):

        data = vmem.data
        shape = data.shape
        data = gaussian_filter1d(data, 2, axis=0)
        is_active = 1
        APD_value = 0
        for i in range(shape[0]):
            data[i, :, :] = filters.gaussian_filter(data[i, :, :], 2)

        super(APDMap, self).__init__(shape[0], shape[1], shape[2])

        self.data = np.zeros_like(data, dtype=np.uint8)
        self.roi = vmem.roi
        self.vmin = 0
        self.vmax = 250
        self.cmap = 'jet'

        for i in range(shape[1]):
            for j in range(shape[2]):
                if self.roi[i][j] != 0:
                    maxtab = []
                    mintab = []
                    x = np.arange(len(data[:, i, j]))
                    v = np.asarray(data[:, i, j])
                    mn, mx = np.Inf, -np.Inf
                    mnpos, mxpos = np.NaN, np.NaN
                    lookformax = True
                    for k in np.arange(len(v)):
                        this = v[k]
                        if this > mx:
                            mx = this
                            mxpos = x[k]
                        if this < mn:
                            mn = this
                            mnpos = x[k]
                        if lookformax:
                            if this < mx-0.08:
                                maxtab.append((mxpos, mx))
                                mn = this
                                mnpos = x[k]
                                lookformax = False
                        else:
                            if this > mn+0.08:
                                mintab.append((mnpos, mn))
                                mx = this
                                mxpos = x[k]
                                lookformax = True
                    maxtab = np.array(maxtab)
                    mintab = np.array(mintab)

                    if len(maxtab) < 4 or len(mintab) < 4:
                          pass
                    else:
                          if maxtab[0, 0] <= mintab[0,0]:
                              APD_start = maxtab[1, 0]
                          else:
                              APD_start = maxtab[0, 0]
                          if maxtab[-1, 0] <= mintab[-1, 0]:
                              APD_end = mintab[-2, 0]
                          else:
                              APD_end = mintab[-1, 0]
                          x_range = np.arange(int(APD_start), int(APD_end), 1)
                          max_convert = interp1d(maxtab[:,0], maxtab[:,1], kind='cubic')
                          min_convert = interp1d(mintab[:,0], mintab[:,1], kind='cubic')
                          max_y = max_convert(x_range)
                          min_y = min_convert(x_range)
                          APD_50 = min_y + (max_y - min_y) * 0.5
                          APD_90 = min_y + (max_y - min_y) * 0.1
                          for k in range(x_range[0], x_range[-1]-1):
                              start_time = filter(lambda x:x<=k, maxtab[:, 0])[-1]
                              if is_active:
                                  if data[k, i, j] >= APD_90[k-x_range[0]] and data[k+1, i, j] < APD_90[k+1-x_range[0]]:
                                      APD_value = k - start_time
                                      is_active = 0
                              else:
                                  if data[k, i, j] <= APD_50[k-x_range[0]] and data[k+1, i, j] > APD_50[k+1-x_range[0]]:
                                      is_active = 1
                              if APD_value > 250:
                                  APD_value = 0
                              self.data[k, i, j] = APD_value
        return


