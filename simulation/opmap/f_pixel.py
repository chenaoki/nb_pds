# -*- coding: utf-8 -*-
import numpy as np
import scipy.interpolate as interpolate
from scipy.ndimage.filters import gaussian_filter1d
from scipy.signal import hilbert
from .f_peakdetect import peakdetect

def f_pixel_peaks(ts):
    try:    
        peaks, bottoms = peakdetect(ts, lookahead=50)
        peaks = np.array(peaks)
        bottoms = np.array(bottoms)
        start = np.array([[0, ts[0]]])
        end = np.array([[len(ts), ts[-1]]])
        peaks_ = np.concatenate((start,peaks, end))
        bottoms_ = np.concatenate((start,bottoms, end))

        f = interpolate.interp1d(peaks_[:,0], peaks_[:,1], kind="linear")
        _peaks_ = f(np.arange(len(ts)))
        f = interpolate.interp1d(bottoms_[:,0], bottoms_[:,1], kind="linear")
        _bottoms_ = f(np.arange(len(ts)))
        
        return (_peaks_, _bottoms_)

    except:
        return (np.ones_like(ts)*np.mean(ts),np.ones_like(ts)*np.mean(ts))
    
def f_pixel_mean(ts, h=0.5):
    peaks, bottoms = f_pixel_peaks(ts)
    return h*peaks+(1-h)*bottoms

def f_pixel_updown(ts, thre_type, thre_up, thre_down):
    
    if thre_type is 'const':
        thre_up_line = np.ones_like(ts) * ( thre_up*ts.max() + (1-thre_up)*ts.min() )
        thre_down_line = np.ones_like(ts) * ( thre_down*ts.max() + (1-thre_down)*ts.min() )
    elif thre_type is 'peak':
        peaks, bottoms = f_pixel_peaks(ts)
        thre_up_line = thre_up*peaks+(1-thre_up)*bottoms
        thre_down_line = thre_down*peaks+(1-thre_down)*bottoms
    
    list_up = []; list_down = [];
    state_up = False
    for i, v in enumerate(ts):
        if i == 0 : continue
        flg_up   = ( ( v > thre_up_line[i] ) and ( not state_up ) )
        flg_down = ( ( v < thre_down_line[i] ) and state_up )
        state_up = ( state_up and (not flg_down) ) or  ( ( not state_up) and flg_up ) 
        if flg_up: list_up.append(i)
        if flg_down: list_down.append(i)
            
    return (list_up, list_down)

def f_pixel_apd(ts, max_beats, thre_type, thre_up, thre_down):
    list_up, list_down = f_pixel_updown(ts, thre_type, thre_up, thre_down)
    ret = np.zeros(max_beats)
    for i in range(max_beats):
        if len(list_up) > i and len(list_down) > i:
            ret[i] = max( 0, list_down[i] - list_up[i] )
    return ret
    
def f_pixel_phase(ts):
    return np.angle(hilbert(ts))

def f_pixel_diff_thre(v, kwargs):
    grad = np.zeros_like(v)
    grad[1:] = v[1:] - v[:-1] 
    t_event = np.where( grad > kwargs['diff_thre'])[0]
    t_inter = t_event[1:] - t_event[:-1]
    to_delete = np.where( t_inter < kwargs['inter_min'])[0] + 1
    t_event = np.delete(t_event, to_delete)
    return t_event
    

