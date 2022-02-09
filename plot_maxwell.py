import os
import math
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

import constants as C
from utils import *

import taichi as ti

real = ti.f64
#ti.init(default_fp=real, arch=ti.cuda)
ti.init(default_fp=real, arch=ti.cpu)
scalar = lambda: ti.field(dtype=real)        # last dimension is scalar

lam_0 = 1.
ell_0 = 1.

batch_size = 2
max_time = int(C.t_sup * C.anal_sr)
n_sample = 2    # number of previous samples as input
n_hidden = 8    # number of hidden units
n_layer  = 1    # number of hidden layers

#------------------------------ 
# taichi variables
#------------------------------ 
tar = scalar()      # target signal    (batch_size, max_time)
inp = scalar()      # input signal     (batch_size, max_time)
byp = scalar()      # bypass signal    (batch_size, max_time)
com = scalar()      # compensated      (batch_size, max_time)
lam = scalar()      # stretch          (batch_size, max_time)
ell_lam = scalar()  # interal stretch  (batch_size, max_time)
ell_dry = scalar()  # interal stretch  (batch_size, max_time)
est = scalar()      # estimate signal  (batch_size, max_time)
dry = scalar()      # uncompensated    (batch_size, max_time)

w_ih = scalar()    # input weight    (n_hidden, n_sample)
b_ih = scalar()    # input bias      (n_hidden)
w_hh = scalar()    # hidden weight   (n_layer, n_hidden, n_hidden)
b_hh = scalar()    # hidden bias     (n_layer, n_hidden)
w_ho = scalar()    # output weight   (1, n_hidden)
b_ho = scalar()    # output bias     (1,)

hid  = scalar()     # hidden state    (batch_size, max_time, n_layer+2, n_hidden)
coef_w = scalar()     # rescaling coefficient
coef_b = scalar()     # rebiasing coefficient

lr = scalar()
loss = scalar()
lam_max = scalar()
dry_max = scalar()
norm_sqr = scalar()
scale = scalar()

#------------------------------ 
# material constants
#------------------------------ 
m_1 = scalar()
m_2 = scalar()
m_3 = scalar()
ma  = scalar()
mb  = scalar()
mc  = scalar()
dt  = scalar()

eps  = scalar()
mu   = scalar()
z_0  = scalar()
beta = scalar()
Vpp  = scalar()
Vdc  = scalar()

def allocate_fields():
    print(">>> allocate fields")
    st = time.time()

    ti.root.dense(ti.i, batch_size).dense(ti.j, max_time).place(tar)
    ti.root.dense(ti.i, batch_size).dense(ti.j, max_time).place(inp)
    ti.root.dense(ti.i, batch_size).dense(ti.j, max_time).place(byp)
    ti.root.dense(ti.i, batch_size).dense(ti.j, max_time).place(com)
    ti.root.dense(ti.i, batch_size).dense(ti.j, max_time).place(lam)
    ti.root.dense(ti.i, batch_size).dense(ti.j, max_time).place(ell_lam)
    ti.root.dense(ti.i, batch_size).dense(ti.j, max_time).place(ell_dry)
    ti.root.dense(ti.i, batch_size).dense(ti.j, max_time).place(est)
    ti.root.dense(ti.i, batch_size).dense(ti.j, max_time).place(dry)

    ti.root.dense(ti.i, n_hidden).dense(ti.j, n_sample+1).place(w_ih, w_ih.grad)    # +1 for time concat
    ti.root.dense(ti.i, n_hidden).place(b_ih, b_ih.grad)
    ti.root.dense(ti.i, n_layer).dense(ti.jk, n_hidden).place(w_hh, w_hh.grad)
    ti.root.dense(ti.i, n_layer).dense(ti.j,  n_hidden).place(b_hh, b_hh.grad)
    ti.root.dense(ti.i, n_hidden).place(w_ho, w_ho.grad)
    ti.root.place(b_ho, b_ho.grad)

    ti.root.dense(ti.i, batch_size).dense(ti.j, max_time).dense(ti.k, n_layer+1).dense(ti.l, n_hidden).place(hid)

    ti.root.place(lr)
    ti.root.place(loss, loss.grad)
    ti.root.place(lam_max, dry_max)
    ti.root.place(coef_w, coef_w.grad)
    ti.root.place(coef_b, coef_b.grad)

    ti.root.place(norm_sqr, scale)
    ti.root.place(eps, mu, z_0, beta, Vpp, Vdc)
    ti.root.place(m_1)
    ti.root.place(m_2)
    ti.root.place(m_3)
    ti.root.place(ma, mb, mc, dt)

    ti.root.lazy_grad()

    tt = round(time.time() - st, 3)
    print(f"... done ({tt} sec)")


#============================== 
# forward subroutine 
#============================== 
@ti.func
def smoothe(t):
    return ti.tanh(float(t / max_time * 8))**2
    #return ti.tanh(float(t / max_time * 100))**2

#@ti.func
def viscous_step_1(b, t):
    Phi = (eps[None] * com[b,t]**2) / (mu[None] * z_0[None]**2)
    Lam = ell_lam[b,t-1]**(-2) * lam[b,t-1]**4 - ell_lam[b,t-1] * lam[b,t-1]

    M  = ma[None] * lam[b,t-1]**4 - mb[None] * lam[b,t-1]
    M += Phi / ma[None]
    M += beta[None] * Lam
    M += mc[None] * lam[b,t-1]**(3/2)
    M *= 1 / (1 + m_1[None] * lam[b,t-1]**3)
    M *= m_2[None] * (dt[None]**2)

    # eq 1
    lam[b,t]  = 3/2 * (lam[b,t-1] - lam[b,t-2])**2
    lam[b,t] *= 1 / (lam[b,t-1] + m_1[None] * lam[b,t-1]**4)
    lam[b,t] += 2*lam[b,t-1] - lam[b,t-2]
    lam[b,t] -= M

#@ti.func
def viscous_step_2(b, t):
    # eq 2
    ell_lam[b,t]  = m_3[None] * dt[None]
    ell_lam[b,t] *= lam[b,t-1]**2 * ell_lam[b,t-1]**(-3) - lam[b,t-1]**(-1) 
    ell_lam[b,t] += ell_lam[b,t-1]

#@ti.func
def bypass_step_1(b, t):
    Phi = (eps[None] * byp[b,t]**2) / (mu[None] * z_0[None]**2)
    Lam = ell_dry[b,t-1]**(-2) * dry[b,t-1]**4 - ell_dry[b,t-1] * dry[b,t-1]

    M  = ma[None] * dry[b,t-1]**4 - mb[None] * dry[b,t-1]
    M += Phi / ma[None]
    M += beta[None] * Lam
    M += mc[None] * dry[b,t-1]**(3/2)
    M *= 1 / (1 + m_1[None] * dry[b,t-1]**3)
    M *= m_2[None] * (dt[None]**2)

    # eq 1
    dry[b,t]  = 3/2 * (dry[b,t-1] - dry[b,t-2])**2
    dry[b,t] *= 1 / (dry[b,t-1] + m_1[None] * dry[b,t-1]**4)
    dry[b,t] += 2*dry[b,t-1] - dry[b,t-2] - M

#@ti.func
def bypass_step_2(b, t):
    # eq 2
    ell_dry[b,t]  = m_3[None] * dt[None]
    ell_dry[b,t] *= dry[b,t-1]**2 * ell_dry[b,t-1]**(-3) - dry[b,t-1]**(-1) 
    ell_dry[b,t] += ell_dry[b,t-1]

#@ti.func
def maxwell_step_1(b, t):
    act = (Vpp[None] + Vdc[None]) * inp[b,t]**.5
    Phi = (eps[None] * act**2) / (mu[None] * z_0[None]**2)
    Lam = ell_lam[b,t-1]**(-2) * lam[b,t-1]**4 - ell_lam[b,t-1] * lam[b,t-1]

    M  = ma[None] * lam[b,t-1]**4 - mb[None] * lam[b,t-1]
    M += Phi / ma[None]
    M += beta[None] * Lam
    M += mc[None] * lam[b,t-1]**(3/2)
    M *= 1 / (1 + m_1[None] * lam[b,t-1]**3)
    M *= m_2[None] * (dt[None]**2)

    # eq 1
    lam[b,t]  = 3/2 * (lam[b,t-1] - lam[b,t-2])**2
    lam[b,t] *= 1 / (lam[b,t-1] + m_1[None] * lam[b,t-1]**4)
    lam[b,t] += 2*lam[b,t-1] - lam[b,t-2] - M

#@ti.func
def maxwell_step_2(b, t):
    # eq 2
    ell_lam[b,t]  = m_3[None] * dt[None]
    ell_lam[b,t] *= lam[b,t-1]**2 * ell_lam[b,t-1]**(-3) - lam[b,t-1]**(-1) 
    ell_lam[b,t] += ell_lam[b,t-1]

@ti.func
def step(lam_curr, lam_prev, ell_curr, actuation):
    Phi  = eps[None] * actuation**2 / (mu[None] * z_0[None]**2)
    Lam  = ell_curr**(-2) * lam_curr**4 - ell_curr * lam_curr

    M  = ma[None] * lam_curr**4 - mb[None] * lam_curr
    M += Phi / ma[None]
    M += beta[None] * Lam
    M += mc[None] * lam_curr**(3/2)
    M *= 1 / (1 + m_1[None] * lam_curr**3)

    # eq 1
    lam_next  = 3/2 * (lam_curr - lam_prev)**2
    lam_next *= 1 / (lam_curr + m_1[None] * lam_curr**4)
    lam_next += 2*lam_curr - lam_prev
    lam_next -= m_2[None] * dt[None]**2 * M

    # eq 2
    ell_next  = m_3[None] * dt[None]
    ell_next *= lam_curr**2 * ell_curr**(-3) - lam_curr**(-1) 
    ell_next += ell_curr

    return lam_next, ell_next

#@ti.kernel
def fdtd_lam(b: ti.i32):
    for t in range(n_sample, max_time):
        viscous_step_1(b,t)
        viscous_step_2(b,t)
#@ti.kernel
def fdtd_dry(b: ti.i32):
    for t in range(n_sample, max_time):
        bypass_step_1(b,t)
        bypass_step_2(b,t)
@ti.kernel
def rescale_lam(b: ti.i32):
    for t in range(n_sample, max_time):
        est[b,t] = (1 - lam[b,t]) * coef_w[None]
@ti.kernel
def rescale_dry(b: ti.i32):
    for t in range(n_sample, max_time):
        dry[b,t] =  1 - dry[b,t]
#++++++++++++++++++++++++++++++ 
#@ti.kernel
#def maxwell(b: ti.i32):
#    for t in range(n_sample, max_time):
#        lam[b,t] = step_lam(lam[b,t-1], lam[b,t-2], ell_lam[b,t-1], (inp[b,t]**.5) * (Vpp[None] + Vdc[None]))
#        ell_lam[b,t] = step_ell(lam[b,t-1], lam[b,t-2], ell_lam[b,t-1], (inp[b,t]**.5) * (Vpp[None] + Vdc[None]))
#        #lam[b,t], ell_lam[b,t] = step(lam[b,t-1], lam[b,t-2], ell_lam[b,t-1], (inp[b,t]**.5) * (Vpp[None] + Vdc[None]))
#++++++++++++++++++++++++++++++ 
#@ti.kernel
def maxwell_1(b: ti.i32):
    for t in range(n_sample, max_time):
        maxwell_step_1(b,t)
#@ti.kernel
def maxwell_2(b: ti.i32):
    for t in range(n_sample, max_time):
        maxwell_step_2(b,t)
#++++++++++++++++++++++++++++++ 

@ti.func
def mlp_i(b: ti.i32, t: ti.i32):
    for i in ti.static(range(n_hidden)):
        act = w_ih[i,0] * (t / max_time)
        #act = 0.
        for j in ti.static(range(1,n_sample+1)):
            act += w_ih[i,j] * inp[b,t-n_sample+j-1]
        act += b_ih[i]
        hid[b,t,0,i] = ti.max(act, 0)
@ti.func
def mlp_h(b: ti.i32, t: ti.i32):
    for i in ti.static(range(n_layer)):
        for j in ti.static(range(n_hidden)):
            act = 0.
            for k in ti.static(range(n_hidden)):
                act += w_hh[i,j,k] * hid[b,t,i,k]
            act += b_hh[i,j]
            hid[b,t,i+1,j] = ti.max(act, 0)
@ti.func
def mlp_o(b: ti.i32, t: ti.i32):
    act = 0.
    for i in ti.static(range(n_hidden)):
        act += w_ho[i] * hid[b,t,n_layer,i]
    act += b_ho[None]

    #============================== 
    #com[b,t] = ti.tanh(act)
    com[b,t] = ti.max(ti.tanh(act), 0)
    #com[b,t] = 1 / (1 + ti.exp(- act))
    #============================== 


@ti.kernel
def compensate(b: ti.i32):
    for t in range(n_sample, max_time):
        mlp_i(b,t)
        mlp_h(b,t)
        mlp_o(b,t)
@ti.kernel
def amplify(b: ti.i32):
    for t in range(n_sample, max_time):
        com[b,t] *= (Vpp[None] + Vdc[None])
        byp[b,t] = inp[b,t] * (Vpp[None] + Vdc[None])
@ti.kernel
def initialize_field():
    for t in range(max_time):
        for b in range(batch_size):
            tar[b, t] -= Vdc[None]
            tar[b, t] *= 1 / Vpp[None]
            inp[b, t] *= smoothe(t) / (Vpp[None] + Vdc[None])
            lam[b, t] = lam_0
            dry[b, t] = lam_0
            ell_lam[b, t] = ell_0
            ell_dry[b, t] = ell_0


#============================== 
# backward subroutines
#============================== 
@ti.kernel
def compute_loss(b: ti.i32):
    for t in range(n_sample,max_time):
        loss[None] += (inp[b,t] - est[b,t])**2 / max_time / batch_size

@ti.kernel
def compute_regs(b: ti.i32):
    for t in range(n_sample,max_time):
        lam_max[None] = ti.max(lam[b,t], lam_max[None])
        dry_max[None] = ti.max(dry[b,t], dry_max[None])
def regularize():
    loss[None] += abs(lam_max[None] - dry_max[None])


#============================== 
# main routine 
#============================== 
def forward(model):
    for b in range(batch_size):
        if model=="visco":
            compensate(b)
            amplify(b)
            fdtd_lam(b)
            fdtd_dry(b)
        elif model=="maxwell":
            amplify(b)
            maxwell_1(b)
            maxwell_2(b)
            fdtd_dry(b)
        else:
            raise NotImplementedError("specify model in ['visco', 'maxwell']")
        rescale_lam(b)
        rescale_dry(b)
    for b in range(batch_size):
        compute_loss(b)
        compute_regs(b)
    regularize()

def load_dict(md,it):
    st = time.time()
    load_path = f'result/{md}/ckpt/{it}.npz'
    print(f">>> load ckpt: {load_path}")
    ckpt = np.load(load_path)
    w_ih.from_numpy(ckpt['w_ih'])
    w_hh.from_numpy(ckpt['w_hh'])
    w_ho.from_numpy(ckpt['w_ho'])
    b_ih.from_numpy(ckpt['b_ih'])
    b_hh.from_numpy(ckpt['b_hh'])
    b_ho.from_numpy(ckpt['b_ho'])
    coef_w.from_numpy(ckpt['coef_w'])
    coef_b.from_numpy(ckpt['coef_b'])
    #lr.from_numpy(ckpt['lr'])
    tt = round(time.time() - st, 3)
    print(f"... done ({tt} sec)")
    return ckpt['it']

def test(args):
    st = time.time()
    print("-"*30)
    print(">>> Routine start")
    print(f"... batch size: {batch_size}")
    print(f"... max time  : {max_time}")
    print(f"... n sample  : {n_sample}")
    print(f"... n hidden  : {n_hidden}")
    print(f"... n layer   : {n_layer}")

    print("-"*30)
    allocate_fields()
    if args.model=='visco':
        assert args.resume is not None, "specify model ckpt to load"
        load_dict(args.model, args.resume)
    else:
        coef_w[None] = 1

    eps[None]  = C.eps
    mu[None]   = C.mu
    z_0[None]  = C.z_0
    beta[None] = C.beta
    Vpp[None]  = C.Vpp
    Vdc[None]  = C.Vdc

    m_1[None] = (8 * C.z_0**2 / C.a_0**2) * C.pre**3
    m_2[None] = (24 * C.mu / (math.pi * C.rho * C.a_0**2)) * C.pre
    m_3[None] = 1 / (C.tau * C.pre)
    ma[None]  = C.pre**2
    mb[None]  = C.pre**(-1)
    mc[None]  = C.pre**(-1) - C.pre**2
    dt[None]  = 1 / C.anal_sr

    print("-"*30)
    print(">>> start test")
    it = time.time()

    # sample data
    wav_np = sample_sine(batch_size, f=args.freq)
    tar.from_numpy(wav_np)
    inp.from_numpy(wav_np)
    initialize_field()

    # run
    forward(args.model)

    # plot
    tt = round(time.time() - it, 3)
    tot_loss = round(loss[None], 5)
    print(f"Loss = {tot_loss} ({tt} sec)")

    inp_np = inp.to_numpy()
    est_np = minmax_normalize(est.to_torch()).numpy()
    dry_np = minmax_normalize(dry.to_torch()).numpy()
    for i in range(batch_size):

        nfft = 2048
        winl = nfft
        hopl = winl // 16

        dry_np[i] *= 1e4
        est_np[i] *= 1e3
        #dry_mag, dry_phs = librosa.magphase(librosa.stft(dry_np[i,n_sample:]))
        #est_mag, est_phs = librosa.magphase(librosa.stft(est_np[i,n_sample:]))
        dry_mag, dry_phs = librosa.magphase(librosa.stft(dry_np[i,n_sample:], n_fft=nfft, hop_length=hopl, win_length=winl, center=False))
        est_mag, est_phs = librosa.magphase(librosa.stft(est_np[i,n_sample:], n_fft=nfft, hop_length=hopl, win_length=winl, center=False))
        f = np.linspace(0, C.anal_sr // 2, est_mag.shape[0]) * 1e-3
        t = np.linspace(0, C.t_sup, est_mag.shape[1])

        plt.figure(figsize=(7,4))
        plt.subplot(121)
        plt.title('Uncompensated')
        librosa.display.specshow(np.log(dry_mag+1e-5), cmap='magma', x_axis='s', y_axis='linear', x_coords=t, y_coords=f)
        plt.colorbar()
        plt.xlabel("time (sec)")
        plt.ylabel("freq (kHz)")
        plt.xticks([0, 0.5, 1])
        plt.yticks([4, 8, 12, 16, 20])
        plt.subplot(122)
        plt.title('Maxwell-compensated')
        librosa.display.specshow(np.log(est_mag+1e-5), cmap='magma', x_axis='s', y_axis='linear', x_coords=t, y_coords=f)
        plt.colorbar()
        plt.xlabel("time (sec)")
        plt.ylabel("freq (kHz)")
        plt.xticks([0, 0.5, 1])
        plt.yticks([4, 8, 12, 16, 20])
        plt.savefig('comparison.png', dpi=500)
        plt.close()


        dry_np[i] *= 1e-4
        dry_max = max(dry_np[i,n_sample:])
        dry_min = min(dry_np[i,n_sample:])
        inp_np[i,n_sample:] *= dry_max - dry_min
        inp_np[i,n_sample:] += dry_min
        t = np.linspace(0,1,dry_np[i,n_sample:].shape[0])
        plt.figure(figsize=(7,4))
        plt.subplot(211)
        plt.title('Actuation voltage')
        plt.plot(t, inp_np[i,n_sample:], 'k-', linewidth=.7, label='input')
        plt.xticks([])
        plt.legend()
        plt.ylabel("$\Phi(t)$")
        plt.subplot(212)
        plt.title('Stretch')
        plt.plot(t, 1 - inp_np[i, n_sample:], 'r--', linewidth=.7, label='ideal')
        plt.plot(t, 1 - dry_np[i, n_sample:], 'k-',  linewidth=.7, label='simulated')
        plt.legend()
        plt.xlabel("$t$ (sec)")
        plt.ylabel("$\lambda(t)$")

        plt.savefig('simulation.png', dpi=500)
        plt.close()



    tt = round(time.time() - st, 3)
    print(f"... done ({tt} sec)")

if __name__=='__main__':
    import argparse
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    #------------------------------ 
    parser.add_argument('--resume', type=int, default=None)
    parser.add_argument('--model', type=str, default='visco')
    #------------------------------ 
    parser.add_argument('--freq', type=float, default=None)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    np.random.seed(args.seed)

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    test(args)


