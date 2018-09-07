import copy
import time
import os, sys
import shutil
import json
import numpy as np

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import art3d
from matplotlib import cm

from opmap.videoData import VideoData
from opmap.vmemMap import VmemMap
from opmap.phaseMap import PhaseMap
from opmap.phaseVarianceMap import PhaseVarianceMap
from elecpy.elecpySession import ElecpySession

from scipy.signal import argrelmax
from scipy.signal import convolve

def phaseComplement(value):
    value -= (value > np.pi)*2*np.pi
    value += (value < - np.pi)*2*np.pi
    return value

def load_sess(path, t_s, t_e, y_s, y_e, x_s, x_e, save_dir=None):

    sess = ElecpySession(path)

    sess.data['tone'] *= (sess.data['tone']>0.0)*1

    for key in sess.data.keys():
        sess.data[key] = sess.data[key][:,y_s:y_e,x_s:x_e]

    #phase map
    cam = VideoData(*sess.data['vmem'].shape)
    cam.data = - sess.data['vmem']
    vmem = VmemMap(cam);del(cam)
    pmap = PhaseMap(vmem, width = vmem.data.shape[2])
    pmap.data = - pmap.data
    if save_dir is not None:
        vmem.saveImage( os.path.join(save_dir, 'vmem'));
        pmap.saveImage( os.path.join(save_dir, 'pmap')); 
    
    sess.data['phase'] = pmap.data
    
    del(vmem);del(pmap)
    
    #cell params
    sess.data['stim'] = -sess.data['cell/xina']#+sess.data['tone']
    sess.data['hj'] = sess.data['cell/h']*sess.data['cell/j']
    sess.data['gate'] = sess.data['hj']*sess.data['cell/m']*sess.data['cell/m']*sess.data['cell/m']
    
    for key in sess.data.keys():
        sess.data[key] = sess.data[key][t_s:t_e,:,:]
        
    return sess

def plot_psum(sess, start=None, end=None):
    
    if start is None: start = 0
    if end is None: end = sess.data['phase'].shape[0]
    
    pmap = sess.data['phase']
    pmap_diff = copy.deepcopy(pmap)

    for f in range(len(pmap_diff)):
        if f == 0: continue
        pmap_diff[f,:,:] = pmap[f,:,:] - pmap[f-1,:,:]

    pmap_diff = phaseComplement(-pmap_diff)

    sess.data['psum'] = np.sum(pmap_diff[start:end], axis=0)
    plt.imshow( sess.data['psum'], cmap='gray', vmin = 0, vmax=2*np.pi)
    

def pixelwise_open(sess):
    L,M,N = sess.data['tone'].shape
    sess.data['open'] = np.zeros_like(sess.data['gate'])
    b = np.ones(10)
    for i in range(M):
        for j in range(N):
            gate = sess.data['gate'][:,i,j]
            gate_conv = convolve(gate, b, 'same')            
            peaks = argrelmax(gate, order=30)[0]            
            for p in peaks:
                sess.data['open'][p:,i,j] = p
            for n in range(L):
                sess.data['open'][n,i,j] *= -1
                sess.data['open'][n,i,j] += n

def plot_phase_surface(sess, frame, flip = False, cut_thre = 2*np.pi*0.75):
    
    fig = plt.figure(figsize=(15,10))
    ax = fig.add_subplot(111, projection='3d')

    _, size_h, size_w = sess.data['phase'].shape

    y = np.arange(size_h)
    x = np.arange(size_w)
    X, Y = np.meshgrid(x, y)
    Z = sess.data['phase'][frame, :, :]

    if flip:
        wire = ax.plot_wireframe(X[::-1], Y, Z)
    else:
        wire = ax.plot_wireframe(X, Y[::-1], Z)
    ax.set_zlim(-np.pi, np.pi)

    # Retrive data from internal storage of plot_wireframe, then delete it
    nx, ny, _  = np.shape(wire._segments3d)
    wire_x = np.array(wire._segments3d)[:, :, 0].ravel()
    wire_y = np.array(wire._segments3d)[:, :, 1].ravel()
    wire_z = np.array(wire._segments3d)[:, :, 2].ravel()
    wire.remove()

    # create data for a LineCollection
    wire_x1 = np.vstack([wire_x, np.roll(wire_x, 1)])
    wire_y1 = np.vstack([wire_y, np.roll(wire_y, 1)])
    wire_z1 = np.vstack([wire_z, np.roll(wire_z, 1)])

    to_delete = np.arange(0, nx*ny, ny)
    wire_x1 = np.delete(wire_x1, to_delete, axis=1)
    wire_y1 = np.delete(wire_y1, to_delete, axis=1)
    wire_z1 = np.delete(wire_z1, to_delete, axis=1)
    scalars = np.delete((wire_z1[0,:] + wire_z1[1,:]) / 2., to_delete)

    segs = [list(zip(xl, yl, zl)) for xl, yl, zl in \
                 zip(wire_x1.T, wire_y1.T, wire_z1.T)]

    # delete false phase discontinuity
    to_delete = np.where( np.array([ abs( seg[0][2] - seg[1][2]) for seg in segs ]) > cut_thre)[0]
    segs = np.delete( np.array(segs), to_delete, axis=0)
    segs = [seg for seg in segs]

    # Plots the wireframe by a  a line3DCollection
    my_wire = art3d.Line3DCollection(segs, cmap="jet")
    my_wire.set_array(scalars)
    ax.add_collection(my_wire)

    plt.colorbar(my_wire)
    
def plot_prs(sess, endframe):
    L,M,N = sess.data['tone'].shape
    b = np.ones(10)
    prs = []
    for y in range(M):
        for x in range(N):
            tone = sess.data['tone'][:endframe,y,x]
            tone_conv = convolve(tone, b, 'same')
            peaks = argrelmax(tone_conv, order=30)[0]
            if len(peaks) > 1:
                t = peaks[-1]
                prs.append([
                    np.max(sess.data['open'][t-5:t+5,y,x]), 
                    tone_conv[t], 
                    sess.data['psum'][y,x], 
                    ((sess.data['psum']>1.*np.pi)*1)[y,x]]
                )   
    prs = np.array(prs)
    sess.data['prs'] = prs
    
    plt.scatter(sess.data['prs'][:,0], sess.data['prs'][:,1], c=sess.data['prs'][:, 2], cmap='jet', vmin=-np.pi, vmax=np.pi)

    #plt.scatter(prs[:,0], prs[:,1], c=prs[:, 3], cmap='jet', vmin=0, vmax=1)
    #plt.xlim([30,70])
    #plt.ylim([0.0, 0.25])
    


    