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
from networks import Model

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

class Compensator(object):
    def __init__(self, args):
        self.lam_0 = 1.
        self.ell_0 = 1.
        self.max_time = int(C.t_sup * C.anal_sr)
        self.n_sample = 2    # number of previous samples as input
        self.batch_size = args.batch_size

        self.model = Model().cuda()

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

        self.spec_loss = SpecLoss(n_fft=1024, win_length=1024, hop_length=256).cuda()
       
        self.optimizer = torch.optim.Adam(
            params=list(self.model.parameters()) \
                  +list(self.spec_loss.parameters()),
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
        shaper = torch.tanh(t * 8)**2
        return x * shaper * shaper.flip(-1)
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
    
    def fdtd(self, lam, ell_lam, inp):
        for t in range(self.n_sample, self.max_time):
            lam[:,t], ell_lam[:,t] = self.step(lam[:,t-1].clone(),
                                               lam[:,t-2].clone(),
                                               ell_lam[:,t-1].clone(),
                                               inp[:,t])
        return lam, ell_lam

    def compensate(self, inp):
        com = self.model(inp.unsqueeze(1))
        return com
    
    def amplify(self, com, inp):
        com = com * (C.Vpp + C.Vdc)
        byp = inp * (C.Vpp + C.Vdc)
        return com, byp
    
    def initialize_field(self, tar, inp):
        tar = tar / (self.Vdc + self.Vpp)
        inp = inp / (self.Vdc + self.Vpp)
        tar = tar.float().cuda()
        inp = inp.float().cuda()
        lam = torch.ones_like(inp).float().cuda() * self.lam_0
        dry = torch.ones_like(inp).float().cuda() * self.lam_0
        ell_lam = torch.ones_like(inp).float().cuda() * self.ell_0
        ell_dry = torch.ones_like(inp).float().cuda() * self.ell_0
        return tar, inp, lam, dry, ell_lam, ell_dry 

    def save_ckpt(self, checkpoint_dir, md, it):
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_state = {
            "model": self.model.state_dict(),
            "loss" : self.spec_loss.state_dict(),
            "optim": self.optimizer.state_dict(),
            "iter" : it,
        }
        checkpoint_path = os.path.join(checkpoint_dir,'{}_{}.pt'.format(md, it))
        torch.save(checkpoint_state, checkpoint_path)

    def load_dict(self, md, it):
        st = time.time()
        load_path = f'result/torch/{md}/ckpt/{it}.npz'
        print(f">>> load ckpt: {load_path}")
        ckpt = np.load(load_path)
        self.model.load_state_dict(ckpt['model'])
        self.spec_loss.load_state_dict(ckpt['loss'])
        self.optimizer.load_state_dict(ckpt['optim']),
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
            com = self.smoothe(com)
            inp = self.smoothe(inp)
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

        est = 1 - lam
        dry = 1 - dry

        com = com / (C.Vpp + C.Vdc)

        #est = est - self.dc_component(est)
        #dry = dry - self.dc_component(dry)

        loss = self.spec_loss(est, inp, guide=True)
        samples = [inp, com, lam, est, dry]

        return loss, samples
    
    def train(self, args):

        os.environ['WANDB_START_METHOD'] = 'thread'
        logger = wandb.init(
            entity="szin",
            project="dea",
            name="fdtd",
        )
        logger.config.update(args)


        LOSS = []
        print("-"*30)
        print(">>> start train")
        self.model.train()
        logger.watch(self.model)
        for i in range(args.n_iter):
            # sample data
            wav_np = sample_batch(self.batch_size, f=args.freq)
            tar = torch.from_numpy(wav_np).float()
            inp = torch.from_numpy(wav_np).float()
            field = self.initialize_field(tar, inp)
            
            loss, samples = self.forward(args.model, field)
            
            self.optimizer.zero_grad()
            loss.backward()
            #print(self.model.fc.bias.grad.data, self.model.fc.bias.data)
            self.optimizer.step()
            
            if i % 1 == 0:
                tot_loss = round(loss.item(), 3)
                LOSS.append(tot_loss)
                avg_loss = round(sum(LOSS) / len(LOSS), 3)
                print(f"... Iter {i};\t Loss = {tot_loss},\t Avg = {avg_loss}")
 
            if i % 5 == 0:
                plots = plot_samples(samples, args.model, i)
                log = {
                    "plots": wandb.Image(plots[0]),
                }
                losses = {
                    "total": tot_loss,
                }
                logger.log({**losses, **log})

            if i % 100 == 0:
                self.save_ckpt(f"result/torch/{args.model}/test/ckpt", args.model, i)
  

    def test(self, args):
        if args.resume is not None:#wav_np = wav_np[:self.max_time]
            load_dict(args.model, args.resume)

        print("-"*30)
        print(">>> start test")
        it = time.time()
    
        self.model.eval()
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
        plot_samples(samples, args.model, 'eval')

    def dc_component(self, signal):
        dry = torch.ones_like(signal) * self.lam_0
        ell_dry = torch.ones_like(signal) * self.ell_0
    
        dry, ell_dry = self.fdtd(dry, ell_dry, signal * (C.Vpp + C.Vdc))
        dry = 1 - dry
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
        samples = [inp, dry, dry, dry, dry]
    
        # plot
        plot_samples(samples, "dry", 'eval')


    def run(self, args):   
        st = time.time()
        print("-"*30)
        print(">>> Routine start")
        print(f"... batch size: {self.batch_size}")
        print(f"... max time  : {self.max_time}")
        print(f"... n sample  : {self.n_sample}")
    
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
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--n_iter', type=int, default=10000000)
    parser.add_argument('--batch_size', type=int, default=128)
    #------------------------------ 

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    compensator = Compensator(args)
    compensator.run(args)


