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

lam_0  = 1.
ell_0  = 1.

batch_size = 8
max_time = int(C.t_sup * C.anal_sr)
n_sample = 2    # number of previous samples as input
n_hidden = 8    # number of hidden units
n_layer  = 1     # number of hidden layers

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
    #tvar = max_time / 10
    #return 1 / (1 + ti.exp(- (t/tvar) + (max_time/tvar) / 2))
    #return ti.sin((np.pi/2) * ti.sin((np.pi/2) * float(t / max_time * 2)))**2
    return ti.tanh(float(t / max_time * 8))**2

@ti.func
def step(lam_curr, lam_prev, ell_curr, actuation):
    Phi  = eps[None] * actuation**2
    Phi *= 1 / (mu[None] * z_0[None]**2)
    Lam  = ell_curr**(-2) * lam_curr**4
    Lam -= ell_curr * lam_curr

    M  = ma[None] * lam_curr**4 - mb[None] * lam_curr 
    M += Phi + beta[None] * Lam
    M += mc[None] * lam_curr**(3/2)
    M *= 1 / (1 + m_1[None] * lam_curr**3)

    # eq 1
    lam_next  = 3/2 * (lam_curr - lam_prev)**2
    lam_next *= 1 / (lam_curr + m_1[None] * lam_curr**4)
    lam_next += 2*lam_curr - lam_prev
    lam_next -= m_2[None] * dt[None]**2 * M

    # eq 2
    ell_next  = m_3[None] * dt[None]
    ell_next *= - lam_curr**(-1) + lam_curr**2 * ell_curr**(-3)
    ell_next += ell_curr

    return lam_next, ell_next

@ti.kernel
def fdtd_lam(b: ti.i32):
    for t in range(n_sample, max_time):
        lam[b,t], ell_lam[b,t] = step(lam[b,t-1], lam[b,t-2], ell_lam[b,t-1], com[b,t])
@ti.kernel
def fdtd_dry(b: ti.i32):
    for t in range(n_sample, max_time):
        dry[b,t], ell_dry[b,t] = step(dry[b,t-1], dry[b,t-2], ell_dry[b,t-1], byp[b,t])
@ti.kernel
def maxwell(b: ti.i32):
    for t in range(n_sample, max_time):
        lam[b,t], ell_lam[b,t] = step(lam[b,t-1], lam[b,t-2], ell_lam[b,t-1], (inp[b,t]**.5) * (Vpp[None] + Vdc[None]))
@ti.kernel
def rescale_lam(b: ti.i32):
    for t in range(n_sample, max_time):
        est[b,t] = (1 - lam[b,t]) * coef_w[None]
@ti.kernel
def rescale_dry(b: ti.i32):
    for t in range(n_sample, max_time):
        dry[b,t] =  1 - dry[b,t]

@ti.func
def mlp_i(b: ti.i32, t: ti.i32):
    for i in ti.static(range(n_hidden)):
        act = w_ih[i,0] * (t / max_time)
        #act = w_ih[i,0] * i
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

#============================== 
# backward subroutines
#============================== 
@ti.kernel
def compute_loss(b: ti.i32):
    for t in range(n_sample,max_time):
        loss[None] += (inp[b,t] - est[b,t])**2 / (max_time-n_sample) / batch_size

@ti.kernel
def compute_regs(b: ti.i32):
    for t in range(n_sample,max_time):
        lam_max[None] = ti.max(lam[b,t], lam_max[None])
        dry_max[None] = ti.max(dry[b,t], dry_max[None])
def regularize():
    loss[None] += abs(lam_max[None] - dry_max[None])

@ti.kernel
def calc_norm():
    for i in range(n_hidden):
        for j in range(n_sample):
            norm_sqr[None] += w_ih.grad[i, j]**2
        norm_sqr[None] += b_ih.grad[i]**2
    for i in range(n_layer):
        for j in range(n_hidden):
            for k in range(n_hidden):
                norm_sqr[None] += w_hh.grad[i, j, k]**2
            norm_sqr[None] += b_hh.grad[i,j]**2
    for i in range(n_hidden):
        norm_sqr[None] += w_ho.grad[i]**2
    norm_sqr[None] += b_ho.grad[None]**2

    #norm_sqr[None] += coef_w.grad[None]**2
    #norm_sqr[None] += coef_b.grad[None]**2

@ti.kernel
def apply_grad():
    for i in range(n_hidden):
        for j in range(n_sample):
            w_ih[i,j] -= scale[None] * w_ih.grad[i, j]
        b_ih[i] -= scale[None] * b_ih.grad[i]
    for i in range(n_layer):
        for j in range(n_hidden):
            for k in range(n_hidden):
                w_hh[i,j,k] -= scale[None] * w_hh.grad[i, j, k]
            b_hh[i,j] -= scale[None] * b_hh.grad[i,j]
    for i in range(n_hidden):
        w_ho[i] -= scale[None] * w_ho.grad[i]
    b_ho[None] -= scale[None] * b_ho.grad[None]

    #print("w_ih", w_ih.grad[0,0])
    #print("w_hh", w_hh.grad[0,0,0])
    #print("w_ho", w_ho.grad[0])
    #print("b_ih", b_ih.grad[0])
    #print("b_hh", b_hh.grad[0,0])
    #print("b_ho", b_ho.grad[None])
    #print("coef", coef_w.grad[None])

    #coef_w[None] -= scale[None] * coef_w.grad[None]
    #coef_b[None] -= scale[None] * coef_b.grad[None]


#============================= 
# initializing subroutines 
#============================== 
@ti.kernel
def init_ih():
    for i in range(n_hidden):
        for j in range(n_sample):
            w_ih[i,j] = ti.random(real) * (2 / (n_sample + n_hidden))**.5
        b_ih[i] = ti.random(real) * (2 / (n_sample + n_hidden))**.5
@ti.kernel
def init_hh():
    for i in range(n_layer):
        for j in range(n_hidden):
            for k in range(n_hidden):
                w_hh[i,j,k] = ti.random(real) * (2 / (n_hidden + n_hidden + n_layer))**.5
            b_hh[i,j] = ti.random(real) * (2 / (n_sample + n_hidden))**.5
@ti.kernel
def init_ho():
    for i in range(n_hidden):
        w_ho[i] = ti.random(real) * (2 / n_hidden)**.5
    b_ho[None] = ti.random(real) * (2 / n_hidden)**.5

def initialize_model():
    print(">>> initialize model")
    st = time.time()
    init_ih()
    init_hh()
    init_ho()
    coef_w[None] = (C.Vdc + C.Vpp) / 5
    #coef_w[None] = 1
    #coef_w[None] = 800
    coef_b[None] = 0.
    tt = round(time.time() - st, 3)
    print(f"... done ({tt} sec)")

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
            maxwell(b)
            fdtd_dry(b)
        else:
            raise NotImplementedError("specify model in ['visco', 'maxwell']")
        rescale_lam(b)
        rescale_dry(b)
        compute_loss(b)
        compute_regs(b)
    regularize()

def backward():
    norm_sqr[None] = 0
    calc_norm()
    #gradient_clip = 0.2
    #scale[None] = lr[None] * min(1.0, gradient_clip / (norm_sqr[None]**0.5 + 1e-4))
    scale[None] = lr[None]
    apply_grad()

def load_dict(it):
    st = time.time()
    load_path = f'result/ckpt/{it}.npz'
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
    lr.from_numpy(ckpt['lr'])
    tt = round(time.time() - st, 3)
    print(f"... done ({tt} sec)")
    return ckpt['it']

def train(args):
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
    if args.resume:
        s_iter = load_dict(args.resume)
    else:
        s_iter = 0
        initialize_model()

    lr[None]  = args.lr

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
    LOSS = []
    avg_loss = np.inf
    prv_loss = np.inf

    print("-"*30)
    print(">>> start iteration")
    print(f"... initial iter  : {s_iter}")
    print(f"... # total iter  : {args.n_iter}")
    for i in range(s_iter, args.n_iter):
        it = time.time()

        # sample data
        wav_np = sample_sine(batch_size, f=args.freq)
        tar.from_numpy(wav_np)
        inp.from_numpy(wav_np)
        initialize_field()

        # run
        with ti.Tape(loss):
            forward(args.model)

        lam_max[None] = 0
        dry_max[None] = 0
        # fit
        backward()

        LOSS.append(loss[None])
        # schedule
        if i % 10000 == 0 and i > 0 and args.schedule:
            avg_loss = round(sum(LOSS[-100:]) / 100, 4)
            if avg_loss >= prv_loss:
                lr[None] *= 0.5
                print(f"{avg_loss} >= {prv_loss}")
            else:
                print(f"{prv_loss} <-- {avg_loss}")
            prv_loss = avg_loss

        # plot
        if i % 1000 == 0:
            tt = round(time.time() - it, 3)
            tot_loss = round(loss[None], 5)
            print(f"Iter {i}/{args.n_iter}  Loss = {tot_loss} ({tt} sec) lr={lr[None]}")

        if i % 5000 == 0:
            os.makedirs(f"result/{args.model}/plot", exist_ok=True)
            for j in range(1):
                plt.figure(figsize=(10,10))
                plt.subplot(421)
                plt.plot(inp.to_numpy()[j,n_sample:], label='inp')
                plt.legend()
                #plt.subplot(423)
                #plt.plot(com.to_numpy()[j,n_sample:], label='com')
                #plt.legend()
                plt.subplot(423)
                plt.plot(lam.to_numpy()[j,n_sample:], label='lam')
                plt.legend()
                plt.subplot(425)
                plt.plot(est.to_numpy()[j,n_sample:], label='est')
                plt.legend()
                plt.subplot(427)
                plt.plot(dry.to_numpy()[j,n_sample:], label='dry')
                plt.legend()

                plt.subplot(422)
                plt.title('inp')
                inp_mag, inp_phs = librosa.magphase(librosa.stft(inp.to_numpy()[j,n_sample:]))
                librosa.display.specshow(np.log(inp_mag+1e-5), cmap='magma')
                plt.colorbar()
                #plt.subplot(422)
                #plt.title('com')
                #com_mag, com_phs = librosa.magphase(librosa.stft(com.to_numpy()[j,n_sample:]))
                #librosa.display.specshow(np.log(com_mag+1e-5), cmap='magma')
                #plt.colorbar()
                plt.subplot(424)
                plt.title('lam')
                lam_mag, lam_phs = librosa.magphase(librosa.stft(lam.to_numpy()[j,n_sample:]))
                librosa.display.specshow(np.log(lam_mag+1e-5), cmap='magma')
                plt.colorbar()
                plt.subplot(426)
                plt.title('est')
                est_mag, est_phs = librosa.magphase(librosa.stft(est.to_numpy()[j,n_sample:]))
                librosa.display.specshow(np.log(est_mag+1e-5), cmap='magma')
                plt.colorbar()
                plt.subplot(428)
                plt.title('dry')
                dry_mag, dry_phs = librosa.magphase(librosa.stft(dry.to_numpy()[j,n_sample:]*1e4))
                librosa.display.specshow(np.log(dry_mag+1e-5), cmap='magma')
                plt.colorbar()
                plt.savefig(f'result/{args.model}/plot/{i}.png')
                plt.close()

        if i % 10000 == 0:
            os.makedirs(f"result/{args.model}/ckpt", exist_ok=True)
            np.savez_compressed(
                f'result/{args.model}/ckpt/{i}',
                it = i,
                lr = lr.to_numpy(),
                w_ih = w_ih.to_numpy(),
                w_hh = w_hh.to_numpy(),
                w_ho = w_ho.to_numpy(),
                b_ih = b_ih.to_numpy(),
                b_hh = b_hh.to_numpy(),
                b_ho = b_ho.to_numpy(),
                coef_w = coef_w.to_numpy(),
                coef_b = coef_b.to_numpy(),
            )
            print(f"... saved checkpoint")

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
    parser.add_argument('--model', type=str, default='visco')
    parser.add_argument('--n_iter', type=int, default=1000000)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--resume', type=int, default=None)
    parser.add_argument('--schedule', type=str2bool, default='true')
    #------------------------------ 
    parser.add_argument('--freq', type=float, default=None)
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()

    np.random.seed(args.seed)
    train(args)


