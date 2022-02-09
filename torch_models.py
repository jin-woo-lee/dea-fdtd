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

import torch
import torch.nn as nn
import torch.nn.functional as F

class Compensator(object):
    def __init__(self, args):
        self.lam_0 = 1.
        self.ell_0 = 1.
        self.max_time = int(C.t_sup * C.anal_sr)
        self.n_sample = 2    # number of previous samples as input
        self.n_hidden = 8    # number of hidden units
        self.batch_size = args.batch_size
        self.mlp = nn.Sequential(
            nn.Linear(self.n_sample+1, self.n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(self.n_hidden),
            nn.Linear(self.n_hidden, self.n_hidden),
            nn.ReLU(),
            nn.BatchNorm1d(self.n_hidden),
            nn.Linear(self.n_hidden, 1),
        )
        self.coef_w = 1
    
        self.eps  = C.eps
        self.mu   = C.mu
        self.z_0  = C.z_0
        self.beta = C.beta
        self.Vpp  = C.Vpp
        self.Vdc  = C.Vdc
    
        self.m_1 = (8 * C.z_0**2 / C.a_0**2) * C.pre**3
        self.m_2 = (24 * C.mu / (math.pi * C.rho * C.a_0**2)) * C.pre
        self.m_3 = 1 / (C.tau * C.pre)
        self.ma  = C.pre**2
        self.mb  = C.pre**(-1)
        self.mc  = C.pre**(-1) - C.pre**2
        self.dt  = 1 / C.anal_sr
       
        self.optimizer = torch.optim.Adam(
            params=list(self.mlp.parameters()),
            lr=args.lr,
            betas=(0.9, 0.999),
            amsgrad=False,
            weight_decay=0.01 ,
        )

    #============================== 
    # forward subroutine 
    #============================== 
    def smoothe(self, x):
        t = torch.ones_like(x).cumsum(-1).float() / self.max_time
        return torch.tanh(t * 8)**2
        #return torch.tanh(t * 100)**2
    
    def step(self, lam_curr, lam_prev, ell_curr, actuation):
        Phi  = self.eps * actuation**2 / (self.mu * self.z_0**2)
        Lam  = ell_curr**(-2) * lam_curr**4 - ell_curr * lam_curr
    
        M  = self.ma * lam_curr**4 - self.mb * lam_curr
        M += Phi / self.ma
        M += self.beta * Lam
        M += self.mc * lam_curr**(3/2)
        M *= 1 / (1 + self.m_1 * lam_curr**3)
    
        # eq 1
        lam_next  = 3/2 * (lam_curr - lam_prev)**2
        lam_next *= 1 / (lam_curr + self.m_1 * lam_curr**4)
        lam_next += 2*lam_curr - lam_prev
        lam_next -= self.m_2 * self.dt**2 * M
    
        # eq 2
        ell_next  = self.m_3 * self.dt
        ell_next *= lam_curr**2 * ell_curr**(-3) - lam_curr**(-1) 
        ell_next += ell_curr
    
        return lam_next, ell_next
    
    def fdtd_lam(self, lam, ell_lam, inp):
        for t in range(self.n_sample, self.max_time):
            lam[:,t], ell_lam[:,t] = self.step(lam[:,t-1], lam[:,t-2], ell_lam[:,t-1], com[:,t])
        return lam, ell_lam
    def fdtd_dry(self, dry, ell_dry, byp):
        for t in range(self.n_sample, self.max_time):
            dry[:,t], ell_dry[:,t] = self.step(dry[:,t-1], dry[:,t-2], ell_dry[:,t-1], byp[:,t])
        return dry, ell_dry
    def rescale_lam(self, lam):
        est = (1 - lam) * self.coef_w
        return est
    def rescale_dry(self, dry):
        dry = 1 - dry
        return dry
    def maxwell(self, lam, ell_lam, inp):
        for t in range(self.n_sample, self.max_time):
            lam[:,t], ell_lam[:,t] = self.step(lam[:,t-1], lam[:,t-2], ell_lam[:,t-1], (inp[:,t]**.5) * (self.Vpp + self.Vdc))
        return lam, ell_lam
    
    def compensate(self, inp):
        com = torch.zeros_like(inp)
        for t in range(self.n_sample, self.max_time):
            tstep = torch.Tensor([t]).tile(inp.size(0)).reshape(-1,1).float()
            input = torch.cat((inp[:,t-self.n_sample:t], tstep), dim=1)
            com[:,t] = self.mlp(input).squeeze(1)
        com = torch.tanh(com)
        com = torch.relu(com)
        return com
    
    def amplify(self, com, inp):
        com *= (self.Vpp + self.Vdc)
        byp = inp * (self.Vpp + self.Vdc)
        return com, byp
    
    def initialize_field(self, tar, inp):
        tar -= self.Vdc
        tar *= 1 / self.Vpp
        inp = self.smoothe(inp) / (self.Vpp + self.Vdc)
        lam = torch.ones_like(inp) * self.lam_0
        dry = torch.ones_like(inp) * self.lam_0
        ell_lam = torch.ones_like(inp) * self.ell_0
        ell_dry = torch.ones_like(inp) * self.ell_0
        return tar, inp, lam, dry, ell_lam, ell_dry 
    
    #============================== 
    # backward subroutines
    #============================== 
    def compute_loss(self, inp, est):
        return F.mse_loss(inp, est)
    
    #============================== 
    # main routine 
    #============================== 
    def load_dict(self, md,it):
        st = time.time()
        load_path = f'result/torch/{md}/ckpt/{it}.npz'
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


    def run(self, args):   
        st = time.time()
        print("-"*30)
        print(">>> Routine start")
        print(f"... batch size: {self.batch_size}")
        print(f"... max time  : {self.max_time}")
        print(f"... n sample  : {self.n_sample}")
        print(f"... n hidden  : {self.n_hidden}")
    
        if args.train:   
            self.train(args)
        if args.test:   
            self.test(args)


    def train(self, args):

        print("-"*30)
        print(">>> start train")
        for i in range(args.n_iter):
            # sample data
            wav_np = sample_sine(self.batch_size, f=args.freq)
            tar = torch.from_numpy(wav_np)
            inp = torch.from_numpy(wav_np)
            field = self.initialize_field(tar, inp)
        
            # run
            loss, samples = self.forward(args.model, field)
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        
            tot_loss = round(loss.item(), 5)
            print(f"Loss = {tot_loss} ({tt} sec)")
 
            # plot
            self.plot_samples(samples)
 

    def test(self, args):
        if args.model=='visco':
            assert args.resume is not None, "specify model ckpt to load"
            load_dict(args.model, args.resume)

        print("-"*30)
        print(">>> start test")
        it = time.time()
    
        # sample data
        wav_np = sample_sine(self.batch_size, f=args.freq)
        tar = torch.from_numpy(wav_np).cuda()
        inp = torch.from_numpy(wav_np).cuda()
        field = self.initialize_field(tar, inp)
    
        # run
        with torch.no_grad():
            loss, samples = self.forward(args.model, field)
    
        tt = round(time.time() - it, 3)
        tot_loss = round(loss.item(), 5)
        print(f"Loss = {tot_loss} ({tt} sec)")
    
        # plot
        self.plot_samples(samples)


    def forward(self, model, field):
        tar, inp, lam, dry, ell_lam, ell_dry  = field
        if model=="visco":
            com = self.compensate(inp)
            com, byp = self.amplify(com, inp)
            lam, ell_lam = self.fdtd_lam(lam, ell_lam, com)
            dry, ell_dry = self.fdtd_dry(dry, ell_dry, byp)
        elif model=="maxwell":
            com = inp
            com, byp = self.amplify(com, inp)
            lam, ell_lam = self.maxwell(lam, ell_lam, inp)
            dry, ell_dry = self.fdtd_dry(dry, ell_dry, byp)
        else:
            raise NotImplementedError("specify model in ['visco', 'maxwell']")
        est = self.rescale_lam(lam)
        dry = self.rescale_dry(dry)
        loss = self.compute_loss(inp, est)
        samples = [inp, lam, est, dry]

        return loss, samples
    

    def plot_samples(self, samples):
        st = time.time()
        inp, lam, est, dry = samples
        est_np = minmax_normalize(est).detach().cpu().numpy()
        dry_np = minmax_normalize(dry).detach().cpu().numpy()
        os.makedirs(f"result/torch/{args.model}/test/plot", exist_ok=True)
        for i in range(self.batch_size):
    
            plt.figure(figsize=(10,10))
            plt.subplot(421)
            plt.plot(inp.detach().cpu().numpy()[i,self.n_sample:], label='inp')
            plt.legend()
            #plt.subplot(423)
            #plt.plot(com.detach().cpu().numpy()[i,self.n_sample:], label='com')
            #plt.legend()
            plt.subplot(423)
            #plt.plot(lam.detach().cpu().numpy()[i,:1000], label='lam')
            plt.plot(lam.detach().cpu().numpy()[i,self.n_sample:], label='lam')
            plt.legend()
            plt.subplot(425)
            ##plt.plot(ell_lam.detach().cpu().numpy()[i,:1000], label='ell')
            #plt.plot(ell_lam.detach().cpu().numpy()[i,self.n_sample:], label='ell')
            #plt.plot(est.detach().cpu().numpy()[i,-10:], label='est')
            #plt.plot(est.detach().cpu().numpy()[i,self.n_sample:], label='est')
            plt.plot(est_np[i,self.n_sample:], label='est')
            plt.legend()
            plt.subplot(427)
            #plt.plot(dry.detach().cpu().numpy()[i,:1000], label='dry')
            #plt.plot(dry.detach().cpu().numpy()[i,self.n_sample:], label='dry')
            #plt.plot(stretch_to_capacitance(dry_np[i,self.n_sample:]), label='dry')
            plt.plot(dry_np[i,self.n_sample:], label='dry')
            plt.legend()
    
            plt.subplot(422)
            plt.title('inp')
            inp_mag, inp_phs = librosa.magphase(librosa.stft(inp.detach().cpu().numpy()[i,self.n_sample:]))
            librosa.display.specshow(np.log(inp_mag+1e-5), cmap='magma')
            plt.colorbar()
            #plt.subplot(422)
            #plt.title('com')
            #com_mag, com_phs = librosa.magphase(librosa.stft(com.detach().cpu().numpy()[i,self.n_sample:]))
            #librosa.display.specshow(np.log(com_mag+1e-5), cmap='magma')
            #plt.colorbar()
            plt.subplot(424)
            plt.title('lam')
            lam_mag, lam_phs = librosa.magphase(librosa.stft(lam.detach().cpu().numpy()[i,self.n_sample:]))
            librosa.display.specshow(np.log(lam_mag+1e-5), cmap='magma')
            plt.colorbar()
            plt.subplot(426)
            plt.title('est')
            #est_mag, est_phs = librosa.magphase(librosa.stft(est.detach().cpu().numpy()[i,self.n_sample:]))
            est_np[i] *= 1e3
            est_mag, est_phs = librosa.magphase(librosa.stft(est_np[i,self.n_sample:]))
            librosa.display.specshow(np.log(est_mag+1e-5), cmap='magma')
            plt.colorbar()
            plt.subplot(428)
            plt.title('dry')
            #dry_mag, dry_phs = librosa.magphase(librosa.stft(dry.detach().cpu().numpy()[i,self.n_sample:]*1e4))
            dry_mag, dry_phs = librosa.magphase(librosa.stft(dry_np[i,self.n_sample:]*1e4))
            librosa.display.specshow(np.log(dry_mag+1e-5), cmap='magma')
            plt.colorbar()
            plt.savefig(f'result/torch/{args.model}/test/plot/{i}.png')
            plt.close()
    
    
            sos = scipy.signal.butter(10, 20, 'hp', fs=C.save_sr, output='sos')
            pt = -10000
            z = 2048
            plt.figure(figsize=(10,10))
            plt.subplot(211)
            plt.title('est')
            est_spec = est_np[i,pt:pt+z]
            est_spec -= np.mean(est_spec)
            est_spec = scipy.signal.sosfilt(sos, est_spec)
            est_spec = np.absolute(np.fft.rfft(est_spec))
            plt.plot(np.log(est_spec))
            plt.subplot(212)
            plt.title('dry')
            dry_spec = dry_np[i,pt:pt+z]
            dry_spec -= np.mean(dry_spec)
            dry_spec = scipy.signal.sosfilt(sos, dry_spec)
            dry_spec = np.absolute(np.fft.rfft(dry_spec))
            plt.plot(np.log(dry_spec))
            plt.savefig(f'result/torch/{args.model}/test/plot/{i}-spec.png')
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
    #------------------------------ 
    parser.add_argument('--train', type=str2bool, default='false')
    parser.add_argument('--test',  type=str2bool, default='false')
    #------------------------------ 
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_iter', type=int, default=1000)
    parser.add_argument('--batch_size', type=int, default=4)
    #------------------------------ 

    args = parser.parse_args()

    np.random.seed(args.seed)
    compensator = Compensator(args)
    compensator.run(args)


