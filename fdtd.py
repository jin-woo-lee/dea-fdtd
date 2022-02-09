import os
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import matplotlib.pyplot as plt
from utils import *

from tqdm import tqdm
import constants as C


# global static constants
m_1 = (8 * C.z_0**2 / C.a_0**2) * C.pre**3
m_2 = (24 * C.mu / (np.pi * C.rho * C.a_0**2)) * C.pre
m_3 = 1 / (C.tau * C.pre)
a = C.pre**2
b = C.pre**(-1)
c = C.pre**(-1) - C.pre**2

def step(lam_curr, lam_prev, ell_curr, phi, func, dt):
    Phi = C.eps * phi / (C.mu * C.z_0**2)
    Lam = ell_curr**(-2) * lam_curr**4 - ell_curr * lam_curr

    M = (a * lam_curr**4 - b * lam_curr + Phi + C.beta * Lam + c * lam_curr**(3/2) ) \
      / (1 + m_1 * lam_curr**3)

    # eq 1
    lam_next = (3/2 * (lam_curr - lam_prev)**2) / (lam_curr + m_1 * lam_curr**4) \
             + 2*lam_curr - lam_prev \
             - m_2 * dt**2 * M

    # eq 2
    ell_next = ell_curr + m_3 * dt * (-lam_curr**(-1) + lam_curr**2 * ell_curr**(-3))

    return lam_next, lam_curr, ell_next


def get_response(t, vol, lam_0, lam_v0, func):
    lam_t  = lam_0
    lam_tt = lam_0
    lam_vt = lam_v0
    LAM = []
    dt = t[1] - t[0]
    for s in range(len(t)):
        phi = vol[s]
        lam_t, lam_tt, lam_vt = step(lam_t, lam_tt, lam_vt, phi, func, dt)
        LAM.append(lam_t)
    return np.array(LAM)

def simulate_sine(freq=1000):
    print("*** Simulate sine")
    dlam_0 = 0.0
    lam_0 = 1.0
    lam_v0 = 1.0
    
    # LaTeX Style
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    t = np.linspace(0,C.t_sup,C.t_sup*C.anal_sr)
    
    f_rel = np.tile(freq, int(C.t_sup*C.anal_sr)) / C.anal_sr
    omega = np.cumsum(2*np.pi*f_rel)
    vol = C.Vdc + C.Vpp * np.sin(omega)
    for j in range(len(vol)):
        vol[j] *= C.smoothe(t[j], T=0.8, func='sinsin')
    
    lam = get_response(t, vol, lam_0, lam_v0, func='sinsin')
    
    name = [
        f'{C.Vdc}_{C.Vpp}',
        f'filt_{C.do_filter}',
        f'{0.8}_{C.t_sup}',
    ]
    name = '-'.join(name)
    filtered = plot(t, name, freq, lam, vol)
    os.makedirs('wave', exist_ok=True)
    #sf.write(filtered, f'wave/{name}.wav', C.anal_sr, subtype='PCM_16')
    print("... done")

def simulate_wav(path):
    print("*** Simulate wav")
    dlam_0 = 0.0
    lam_0 = 1.0
    lam_v0 = 1.0
    
    # LaTeX Style
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    wav, sr = librosa.load(path)
    if sr != C.anal_sr:
        print(f"... original sr: {sr}")
        print(f"... resample to: {C.anal_sr}")
        wav = librosa.resample(wav, sr, C.anal_sr)
    if len(wav.shape) > 1 and wav.shape[-1] > 1:
        print(f"... handle stereo to mono")
        wav = wav.sum(-1)
    C.t_sup = len(wav) / C.anal_sr
    t = np.linspace(0,C.t_sup,int(C.t_sup*C.anal_sr))
    
    vol = C.Vdc + C.Vpp * wav
    for j in range(len(vol)):
        vol[j] *= C.smoothe(t[j], T=0.8, func='sinsin')
    
    lam = get_response(t, vol, lam_0, lam_v0, func='sinsin')
    
    fname = path.split('/')[-1].split('.')[0]
    name = [
        f'{C.Vdc}_{C.Vpp}',
        f'filt_{C.do_filter}',
        f'{fname}',
    ]
    name = '-'.join(name)
    freq = None
    plot(t, name, freq, lam, vol)
    os.makedirs('wave', exist_ok=True)
    #sf.write(filtered, f'wave/{name}.wav', C.anal_sr, subtype='PCM_16')
    print("... done")


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
    parser.add_argument('--sine', type=str2bool, default='false')
    parser.add_argument('--wav', type=str2bool, default='false')
    parser.add_argument('--freq', type=float, default=1000.)
    parser.add_argument('--wav_path', type=str, default="./librispeech-sample.wav")
    #------------------------------ 
    args = parser.parse_args()
    if args.sine:
        simulate_sine(args.freq)
    if args.wav:
        simulate_wav(args.wav_path)

