import numpy as np
from glob import glob
import cv2
from .videoData import VideoData

class RawCam( VideoData ):
    cam_dtype={
        'sa4':np.ushort,
        'mini':np.ushort,
        'max':np.uint8,
        'max10':np.ushort
    }

    def __init__(self, path, cam_type, image_width, image_height, frame_start, frame_end):

        if "numpy" == cam_type:

            self.files = sorted(glob(path+"/vmem_*.npy"))
            assert len(self.files) > 0
            if frame_end < 0 : frame_end = len(self.files) + frame_end + 1
            self.files = self.files[frame_start:frame_end]

            super(RawCam, self).__init__(len(self.files), image_height, image_width)

            for i, f in enumerate(self.files):
                im = np.load(f)
                self.data[i, :,:] = im

        else:

            self.files = sorted(glob(path+"/*.raw*"))
            assert len(self.files) > 0
            self.files = self.files[frame_start:frame_end]

            super(RawCam, self).__init__(len(self.files), image_height, image_width)

            for i, f in enumerate(self.files):
                im = np.fromfile(f, dtype=self.cam_dtype[cam_type])
                im = im.reshape(image_height, image_width)
                self.data[i, :,:] = im

        self.vmin = np.min(self.data)
        self.vmax = np.max(self.data)
        self.cmap = 'gray'

        return

    def selectPoints(self, savepath="./selectPoints.png"):

        points = []
        def onClick(event, x, y, flag, params):
            wname, img = params
            if event == cv2.EVENT_LBUTTONDOWN:
                img_disp = np.copy(img)
                points.append((x, y))
                for i, p in enumerate(points):
                    cv2.circle(img_disp, p, 2, (255,0,0))
                    cv2.putText(img_disp,str(i),(p[0]-5, p[1]-5),cv2.FONT_HERSHEY_PLAIN, 0.6,(255,0,0))
                    cv2.imshow(wname, img_disp)
                    cv2.imwrite(savepath, img_disp)

        wname = "selectPoints"
        img = self.data[0,:,:]

        assert self.vmin < self.vmax
        img = (255 * (img - self.vmin) / float(self.vmax - self.vmin))
        img = img.astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        cv2.namedWindow(wname)
        cv2.setMouseCallback(wname, onClick, [wname, img] )
        cv2.imshow(wname, img)
        while cv2.waitKey(0) != 27 : pass
        cv2.destroyWindow(wname)
        return np.array(points)


