import os
import numpy as np

def phase_add(a, b):
    a_ = np.cos(a) + np.sin(a)*1j
    b_ = np.cos(b) + np.sin(b)*1j
    return np.angle(a_*b_)

def phase_mean(X):
    a = np.mean( np.cos(X) + 1j*np.sin(X))
    return np.angle(a)

def phaseComplement(value):
    value -= (value > np.pi)*2*np.pi
    value += (value < - np.pi)*2*np.pi
    return value

def phase_variance(X):
    a = np.mean( np.cos(X) + 1j*np.sin(X))
    return 1-np.abs(a)

def makeMovie(path, img_type='png'):

    #cmd = 'ffmpeg -r 15 -y -i "{0}/%06d.{1}" -c:v libx264 -pix_fmt yuv420p -qscale 0 "{0}.avi"'.format(path, img_type)
    cmd = 'ffmpeg -r 15 -y -i "{0}/%06d.{1}" -vcodec rawvideo "{0}.avi"'.format(path, img_type)
    print(cmd)
    os.system(cmd)

