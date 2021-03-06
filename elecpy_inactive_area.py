#!/usr/local/bin/python

import os, sys
import json
import numpy as np
import matplotlib.pyplot as plt
import chainer
from chainer import cuda
from matplotlib import animation
from optparse import OptionParser

from solver.PDE import PDE
from stim.ExtracellularStimulator import ExtracellularStimulator
from stim.MembraneStimulator import MembraneStimulator
from cell.ohararudy.model import model as cell_model_ohararudy
from cell.luorudy.model import model as cell_model_luorudy
from cell.mahajan.model import model as cell_model_mahajan
from util.cmap_bipolar import bipolar

# global variables
sim_params = None
cells = None
stims_ext = []
stims_mem = []
i_ion    = None
phie     = None
i_ext_e  = None
i_ext_i  = None
rhs_phie = None 
rhs_vmem = None 
vmem     = None 

# Functions
def conv_cntSave2time(cnt_save):
    global sim_params
    udt          = sim_params['time']['udt']     # Universal time step (ms)
    cnt_log      = sim_params['log']['cnt']      # num of udt for logging
    return udt*cnt_log

def conv_cntUdt2time(cnt_udt):
    global sim_params
    udt          = sim_params['time']['udt']     # Universal time step (ms)
    return cnt_udt * udt

def conv_time2cntUdt(t):
    global sim_params
    udt          = sim_params['time']['udt']     # Universal time step (ms)
    return int(t/udt)

def conv_time2cntSave(t):
    global sim_params
    udt          = sim_params['time']['udt']     # Universal time step (ms)
    cnt_log      = sim_params['log']['cnt']      # num of udt for logging
    return conv_time2cntUdt(t) // cnt_log 

def sim_generator( params ):

    global sim_params, cells, stims_ext, stims_mem
    global i_ion, phie, i_ext_e, i_ext_i, rhs_phie, rhs_vmem, vmem
    
    sim_params = params

    assert sim_params is not None

    print "elecpy simulation start!"

    cuda.get_device(0).use()

    # Constants
    Sv           = 1400                  # Surface-to-volume ratio (cm^-1)
    Cm           = 1.0                   # Membrane capacitance (uF/cm^2)
    sigma_l_i    = 1.74                  # (mS/cm)
    sigma_t_i    = 0.19                  # (mS/cm)
    sigma_l_e    = 6.25                  # (mS/cm)
    sigma_t_e    = 2.36                  # (mS/cm)

    # Geometory settings
    im_h         = sim_params['geometory']['height']
    im_w         = sim_params['geometory']['width']
    ds           = sim_params['geometory']['ds'] # Spatial discretization step (cm)
    N            = im_h*im_w

    # Time settings
    udt          = sim_params['time']['udt']     # Universal time step (ms)
    time_end     = sim_params['time']['end']

    # Logging settings
    cnt_log      = sim_params['log']['cnt']      # num of udt for logging
    savepath     = sim_params['log']['path']

    # Cell model settings
    if sim_params['cell_type'] == 'ohararudy':
        cells = cell_model_ohararudy((N))
    if sim_params['cell_type'] == 'luorudy':
        cells = cell_model_luorudy((N))
    if sim_params['cell_type'] == 'mahajan':
        cells = cell_model_mahajan((N))
    assert cells is not None

    print "Stimulation settings",
    stims_ext = []
    stims_mem = []
    if 'stimulation' in sim_params.keys():
        stim_param = sim_params['stimulation']
        if 'extracellular' in stim_param:
            for param in stim_param['extracellular']:
                stim = ExtracellularStimulator(**param)
                assert tuple(stim.shape) == (im_h, im_w)
                stims_ext.append(stim)
        if 'membrane' in stim_param:
            for param in stim_param['membrane']:
                stim = MembraneStimulator(**param)
                assert tuple(stim.shape) == (im_h, im_w)
                stims_mem.append(stim)
    print "...done"

    print "Allocating data...",
    cells.create()
    i_ion              = np.zeros((N),dtype=np.float64)
    phie               = np.zeros((N),dtype=np.float64)
    i_ext_e            = np.zeros((N),dtype=np.float64)
    i_ext_i            = np.zeros((N),dtype=np.float64)
    rhs_phie           = np.zeros((N),dtype=np.float64)
    rhs_vmem           = np.zeros((N),dtype=np.float64)
    tone               = np.zeros((N),dtype=np.float64)
    vmem               = np.copy(cells.get_param('v'))
    
    mask_ion = np.ones((im_h, im_w), dtype=np.float64)
    for h in range(im_h):
        for w in range(im_w):
            distance = (h-im_h//2)**2 + (w-im_w//2)**2
            if distance < ( (5**2)*2 ):
                mask_ion[h,w] = 0.
    mask_ion = mask_ion.flatten()
    
    print "...done"

    print "Initializing data...",
    if 'restart' in sim_params.keys():
        cnt_restart = sim_params['restart']['count']
        srcpath = sim_params['restart']['source']
        pfx = '_{0:0>4}'.format(cnt_restart)
        phie = np.load('{0}/phie{1}.npy'.format(srcpath,pfx)).flatten()
        vmem = np.load('{0}/vmem{1}.npy'.format(srcpath,pfx)).flatten()
        cells.load('{0}/cell{1}'.format(srcpath,pfx))
        cnt_udt = cnt_restart * cnt_log
    print "...done"

    print 'Building PDE system ...',
    sigma_l      = sigma_l_e + sigma_l_i
    sigma_t      = sigma_t_e + sigma_t_i
    pde_i = PDE( im_h, im_w, sigma_l_i, sigma_t_i, ds )
    pde_m = PDE( im_h, im_w, sigma_l,   sigma_t,   ds )
    print '...done'

    # Initialization
    t         = 0.                       # Time (ms)
    cnt_udt   = 0                        # Count of udt
    dstep     = 1                        # Time step (# of udt)
    cnt_save  = -1
    
    run_udt   = True                     # Flag of running sim in udt
    flg_st    = False                    # Flaf of stimulation
    cnt_st_off = 0

    print 'Main loop start!'
    while t < time_end:

        t = conv_cntUdt2time(cnt_udt)
        dt = dstep * udt

        # Stimulation control
        i_ext_e[:] = 0.0
        flg_st_temp = False
        for s in stims_ext:
            i_ext_e += s.get_current(t)*Sv
            flg_st_temp = flg_st_temp or s.get_flag(t)
        for s in stims_mem:
            cells.set_param('st', s.get_current(t)) 

        # step.1 cell state transition
        cells.set_param('dt', dt )
        cells.set_param('v', cuda.to_gpu(vmem) )
        cells.update()
        i_ion = cells.get_param('it')
        
        i_ion = i_ion * mask_ion
        
        # step.2 phie
        rhs_phie = i_ext_e - i_ext_i - pde_i.forward(vmem)
        pde_cnt, phie = pde_m.solve(phie, rhs_phie, tol=1e-2, maxcnt=1e5)
        phie -= phie[0]

        # step.3 vmem
        rhs_vmem = pde_i.forward(vmem)
        rhs_vmem += pde_i.forward(phie)
        tone     = ( rhs_vmem * dt ) / (Cm * Sv)
        rhs_vmem -= i_ion * Sv
        rhs_vmem += i_ext_i
        rhs_vmem *= 1 / (Cm * Sv)
        vmem += dt * rhs_vmem

        # Logging & error check
        cnt_save_now = conv_time2cntSave(t)
        if cnt_save_now != cnt_save:
            cnt_save = cnt_save_now
            sys.stdout.write('\r------------------{0}/{1}ms'.format(t, time_end))
            sys.stdout.flush()
            np.save('{0}/phie_{1:0>4}'.format(savepath,cnt_save), phie.reshape((im_h, im_w)))
            np.save('{0}/vmem_{1:0>4}'.format(savepath,cnt_save), vmem.reshape((im_h, im_w)))
            np.save('{0}/tone_{1:0>4}'.format(savepath,cnt_save), tone.reshape((im_h, im_w)))
            cells.save('{0}/cell_{1:0>4}'.format(savepath,cnt_save))
            yield vmem

            flg = False
            for i,v in enumerate(vmem):
                if v != v :
                    print "error : invalid value {1} @ {0} ms, index {2}".format(t, v, i)
                    flg = True
                    break
            if flg is True:
                break

        # Stim off count
        if flg_st_temp is False:
            if flg_st is True:
                cnt_st_off = 0
            else:
                cnt_st_off += 1
            flg_st = flg_st_temp

        # Time step control
        if run_udt:
            if cnt_st_off >= 3 and cnt_udt % 10 == 0:
                dstep = 2
                run_udt = False
        else:
            if pde_cnt > 5:
                dstep = 1
                run_udt = True

        cnt_udt += dstep

    print "elecpy done"
    yield False

if __name__ == '__main__':

    parser = OptionParser()
    parser.add_option(
        '-p','--param_file',
        dest='param_file', action='store', type='string', default='./temp/sim_params.json',
        help="json file of simulation parameters")
    parser.add_option(
        '-d','--dst',
        dest='savepath', action='store', type='string', default='./temp/result/',
        help="Save data path.")

    (options, args) = parser.parse_args()

    with open (options.param_file,'r') as f:
        sim_params = json.load(f)
    if not os.path.isdir(options.savepath) :
        os.mkdir(options.savepath)
    with open('{0}/sim_params.json'.format(options.savepath), 'w') as f:
        json.dump(sim_params, f, indent=4)
    sim_params['log']['path'] = options.savepath

    im_h         = sim_params['geometory']['height']
    im_w         = sim_params['geometory']['width']
    fig = plt.figure(figsize=(5,5))
    im = plt.imshow(
        np.zeros((im_h,im_w),dtype=np.float64),
        vmin = -100.0, vmax = 100.0,
        cmap=bipolar(neutral=0, lutsize=1024),
        interpolation='nearest')
    plt.axis('off')

    g = sim_generator()

    def init():
        im.set_array(np.zeros((im_h,im_w),dtype=np.float64))
        return (im,)

    def draw(data):
        try:
            vmem = g.next()
            im.set_array(vmem)
            return (im,)
        except StopIteration:
            return init()

    anim = animation.FuncAnimation(
            fig, draw, init_func=init, 
            save_count = conv_time2cntSave(sim_params['time']['end']),
            blit=False, interval=50, repeat=False)
    plt.show()

