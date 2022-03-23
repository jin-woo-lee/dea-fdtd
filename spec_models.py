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
from loss import SpecLoss

import torch
import torch.nn as nn
import torch.nn.functional as F

class Compensator(object):
    def __init__(self, args):
        self.lam_0 = 1.
        self.ell_0 = 1.
        self.max_time = int(C.t_sup * C.anal_sr)
        self.n_sample = 2    # number of previous samples as input
        self.n_hidden = 64   # number of hidden units
        self.gain = 1e3
        self.batch_size = args.batch_size

        nch = 32
        ker = 5
        stride = 1
        padding = 4
        dilation = 2
        self.model = nn.Sequential(
            nn.Conv1d(1, nch, ker, stride=stride, padding=padding, dilation=dilation),
            nn.LeakyReLU(negative_slope=0.3),
            nn.BatchNorm1d(nch),
            #------------------------------ 
            nn.Conv1d(nch, nch, ker, stride=stride, padding=padding, dilation=dilation),
            nn.LeakyReLU(negative_slope=0.3),
            nn.BatchNorm1d(nch),
            #------------------------------ 
            nn.Conv1d(nch, 1, 1, bias=False),
        ).cuda()

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
            params=list(self.model.parameters()),
            lr=args.lr,
            betas=(0.9, 0.999),
            amsgrad=False,
            weight_decay=0.01 ,
        )

        self.spec_loss = SpecLoss(n_fft=1024, win_length=1024, hop_length=256).cuda()

    #============================== 
    # forward subroutine 
    #============================== 
    def smoothe(self, x):
        t = torch.ones_like(x).cumsum(-1).float() / self.max_time
        return x * torch.tanh(t * 8)**2
        #return torch.tanh(t * 100)**2
    
    def step(self, lam_curr, lam_prev, ell_curr, actuation):
        Phi  = self.eps * actuation**2 / (self.mu * self.z_0**2)
        Lam  = ell_curr**(-2) * lam_curr**4 - ell_curr * lam_curr
    
        M = self.ma * lam_curr**4 - self.mb * lam_curr + Phi / self.ma
        M = M + self.beta * Lam + self.mc * lam_curr**(3/2)
        M = M / (1 + self.m_1 * lam_curr**3)
    
        # eq 1
        lam_next = (3/2 * (lam_curr - lam_prev)**2) / (lam_curr + self.m_1 * lam_curr**4) \
                 + 2*lam_curr - lam_prev - self.m_2 * self.dt**2 * M
    
        # eq 2
        ell_next = self.m_3 * self.dt * (lam_curr**2 * ell_curr**(-3) - lam_curr**(-1)) + ell_curr
    
        return lam_next, ell_next
    
    #def fdtd(self, lam, ell_lam, inp):
    #    for t in range(self.n_sample, self.max_time):
    #        lam[:,t], ell_lam[:,t] = self.step(lam[:,t-1], lam[:,t-2], ell_lam[:,t-1], inp[:,t])
    #        #lam[:,t] = lam[:,t-1] * inp[:,t]
    #    return lam, ell_lam

    def fdtd(self, lam, ell_lam, inp):
        lam_t0 = F.pad(lam, (0,2), value=1.)
        inputs = F.pad(inp, (2,0))
        ell_t0 = F.pad(ell_lam, (0,2), value=1.)
        t_mask = torch.zeros_like(lam_t0)
        for t in range(self.n_sample, self.max_time):
            #lam[:,t], ell_lam[:,t] = self.step(lam[:,t-1], lam[:,t-2], ell_lam[:,t-1], inp[:,t])
            lam_t2 = lam_t0.roll(2,-1)    # lam_{t-2}
            lam_t1 = lam_t0.roll(1,-1)    # lam_{t-1}
            ell_t1 = ell_t0.roll(1,-1)    # lam_{t-1}

            t_mask[:,t] = 1
            lam, ell = self.step(lam_t2, lam_t1, ell_t1, inputs)
            lam_t0 = lam_t0 + t_mask * lam - t_mask
            ell_t0 = ell_t0 + t_mask * ell - t_mask
            t_mask[:,t] = 0

        return lam, ell_lam

    def rescale_lam(self, lam):
        est = (1 - lam) * self.coef_w
        return est
    def rescale_dry(self, dry):
        dry = 1 - dry
        return dry
    
    def compensate(self, inp):
        #com = self.model(inp)
        com = self.model(inp.unsqueeze(1))
        com = torch.sigmoid(com)
        return com
    
    def amplify(self, com, inp):
        #com = com * (self.Vpp + self.Vdc)
        #byp = inp * (self.Vpp + self.Vdc)
        com = com * self.gain
        byp = inp * self.gain
        return com, byp
    
    def initialize_field(self, tar, inp):
        tar = tar - self.Vdc
        tar = tar / self.Vpp
        tar = tar.float().cuda()
        inp = self.smoothe(inp / (self.Vpp + self.Vdc)).float().cuda()
        lam = torch.ones_like(inp).float().cuda() * self.lam_0
        dry = torch.ones_like(inp).float().cuda() * self.lam_0
        ell_lam = torch.ones_like(inp).float().cuda() * self.ell_0
        ell_dry = torch.ones_like(inp).float().cuda() * self.ell_0
        return tar, inp, lam, dry, ell_lam, ell_dry 

    
    #============================== 
    # utils
    #============================== 
    def plot_samples(self, samples, model):
        st = time.time()
        inp, lam, est, dry = samples
        est_np = minmax_normalize(est).detach().cpu().numpy()
        dry_np = minmax_normalize(dry).detach().cpu().numpy()
        os.makedirs(f"result/torch/{model}/test/plot", exist_ok=True)
        for i in range(len(inp)):
    
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
            plt.savefig(f'result/torch/{model}/test/plot/{i}.png')
            plt.close()
    
    
            #sos = scipy.signal.butter(10, 100, 'hp', fs=C.save_sr, output='sos')
            #pt = -10000
            #z = 2048
            #plt.figure(figsize=(10,10))
            #plt.subplot(211)
            #plt.title('est')
            #est_s = est_np[i,pt:pt+z]
            #est_s -= np.mean(est_s)
            #est_s = scipy.signal.sosfilt(sos, est_s)
            #est_spec = np.absolute(np.fft.rfft(est_s))
            #plt.plot(np.log(est_spec))
            #plt.subplot(212)
            #plt.title('dry')
            #dry_s = dry_np[i,pt:pt+z]
            #dry_s -= np.mean(dry_s)
            #dry_s = scipy.signal.sosfilt(sos, dry_s)
            #dry_spec = np.absolute(np.fft.rfft(dry_s))
            #plt.plot(np.log(dry_spec))
            #plt.savefig(f'result/torch/{model}/test/plot/{i}-spec.png')
            #plt.close()

            lens = 10000
            os.makedirs(f'result/torch/{model}/test/wave', exist_ok=True)
            est_w = rms_normalize(est_np[i,lens:])
            dry_w = rms_normalize(dry_np[i,lens:])
            sf.write(f'result/torch/{model}/test/wave/{i}-est.wav', est_w, C.anal_sr, subtype='PCM_16')
            sf.write(f'result/torch/{model}/test/wave/{i}-dry.wav', dry_w, C.anal_sr, subtype='PCM_16')
            plt.figure(figsize=(10,10))
            plt.subplot(211)
            plt.plot(est_np[i,lens:])
            plt.subplot(212)
            plt.plot(dry_np[i,lens:])
            plt.savefig(f'result/torch/{model}/test/wave/{i}.png')
            plt.close()
    
        tt = round(time.time() - st, 3)
        print(f"... done ({tt} sec)")

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

    
    #============================== 
    # main routine 
    #============================== 
    def forward(self, model, field):
        tar, inp, lam, dry, ell_lam, ell_dry  = field
        if model=="visco":
            com = self.compensate(inp)
            com, byp = self.amplify(com, inp)
            com = com.squeeze(1)
            lam, ell_lam = self.fdtd(lam, ell_lam, com)
            dry, ell_dry = self.fdtd(dry, ell_dry, byp)
        elif model=="maxwell":
            com = inp ** .5
            com = minmax_normalize(com, inp)
            com, byp = self.amplify(com, inp)
            lam, ell_lam = self.fdtd(lam, ell_lam, com)
            dry, ell_dry = self.fdtd(dry, ell_dry, byp)
        else:
            raise NotImplementedError("specify model in ['visco', 'maxwell']")
        est = self.rescale_lam(lam)
        dry = self.rescale_dry(dry)

        #est = est - self.dc_component(est)
        #dry = dry - self.dc_component(dry)

        #loss = F.mse_loss(est, inp)
        print(est.shape)
        print(inp.shape)
        #loss = self.spec_loss(est, inp)
        loss = F.mse_loss(est, inp)
        samples = [inp, lam, est, dry]

        return loss, samples
    
    def train(self, args):

        LOSS = []
        print("-"*30)
        print(">>> start train")
        for i in range(args.n_iter):
            # sample data
            wav_np = sample_batch(self.batch_size, f=args.freq)
            tar = torch.from_numpy(wav_np).float()
            inp = torch.from_numpy(wav_np).float()
            field = self.initialize_field(tar, inp)
            
            loss, samples = self.forward(args.model, field)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if i % 10 == 0:
                tot_loss = round(loss.item(), 5)
                LOSS.append(tot_loss)
                avg_loss = round(sum(LOSS) / len(LOSS), 5)
                #print(f"Iter {i};\t Loss = {tot_loss},\t Avg = {avg_loss}")
                #print(f"byp {tar[0]}")
                #print(f"com {com[0]}")
                #print(f"tar {tar[0]}")
                #print(f"est {est[0]}")
                #print("="*30)
 
       # plot
       #self.plot_samples(samples)
  

    def test(self, args):
        if args.resume is not None:#wav_np = wav_np[:self.max_time]
            load_dict(args.model, args.resume)

        print("-"*30)
        print(">>> start test")
        it = time.time()
    
        # sample data
        if args.data is not None:
            wav_np, sr = sf.read(args.data)
            print(f"... Loaded {args.data}")
            print(f"... shape {wav_np.shape}")
            if len(wav_np.shape) > 1:
                if  wav_np.shape[0] > wav_np.shape[1]:
                    wav_np = np.mean(wav_np, axis=1)
                elif  wav_np.shape[0] < wav_np.shape[1]:
                    wav_np = np.mean(wav_np, axis=0)
                print(f"... mixdown to {wav_np.shape}")
            if sr != C.anal_sr:
                #wav_np = librosa.resample(wav_np, sr, C.anal_sr) #print(f"... resampled sr {sr} to {C.anal_sr}")
                C.anal_sr = sr
            if wav_np.shape[0] > self.max_time:
                #wav_np = wav_np[:self.max_time]
                #print(f"... trimmed to max length {self.max_time}")
                self.max_time = len(wav_np)
            wav_np = C.Vdc + C.Vpp * wav_np
            wav_np = np.expand_dims(wav_np, axis=0)
        else:
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
        self.plot_samples(samples, args.model)

    def dc_component(self, signal):
        dry = torch.ones_like(signal) * self.lam_0
        ell_dry = torch.ones_like(signal) * self.ell_0
    
        dry, ell_dry = self.fdtd(dry, ell_dry, signal * self.gain)
        dry = self.rescale_dry(dry)
        return dry

    def dry(self, args):
        print("-"*30)
        print(">>> start dry run")
        it = time.time()
    
        # sample data
        wav_np = sample_sine(self.batch_size, f=args.freq)
        tar = torch.from_numpy(wav_np).cuda()
        inp = torch.from_numpy(wav_np).cuda()
        inp = self.smoothe(inp / (self.Vpp + self.Vdc))

        # run
        dry = self.dc_component(inp)
    
        tt = round(time.time() - it, 3)
        samples = [inp, dry, dry, dry]
    
        # plot
        self.plot_samples(samples, "dry")


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
        if args.dry:   
            self.dry(args)


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
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--freq', type=float, default=None)
    parser.add_argument('--seed', type=int, default=0)
    #------------------------------ 
    parser.add_argument('--train', action="store_true", help="when this flag is given, train model")
    parser.add_argument('--test',  action="store_true", help="when this flag is given, evaluate model")
    parser.add_argument('--dry',  type=str2bool, default='false')
    #------------------------------ 
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--n_iter', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=4)
    #------------------------------ 

    args = parser.parse_args()

    np.random.seed(args.seed)
    compensator = Compensator(args)
    compensator.run(args)


