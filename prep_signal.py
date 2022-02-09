import numpy as np
import torch
import soundfile as sf
import librosa

def sine(n_samples, freq, sr, device='cuda'):
    t = torch.arange(n_samples).to(device) / sr
    f_rel = torch.Tensor(freq).tile(n_samples) / sr
    omega = 2*np.pi*f_rel.cumsum(0)
    return torch.sin(omega)

def smoothe(n_samples, sr, device='cuda'):
    t = torch.arange(n_samples).to(device)
    #return torch.tanh(t / n_samples * 50)**2
    z = t / n_samples * 50
    return ( (e**z - e**(-z)) / (e**z + e**(-z)) ) ** 2

def prepare_pilot_signal(args):
    n_samples = int(args.sr * args.time)

    dc = 0.4  * torch.ones(n_samples).cuda()
    ac = 0.02 * sine(n_samples, [100], args.sr, dc.device)
    sm = smoothe(n_samples, args.sr, dc.device)

    dc = dc.cpu().numpy()
    ac = ac.cpu().numpy()
    sm = sm.cpu().numpy()

    sf.write('dc.wav', dc * sm, args.sr, subtype='PCM_16')
    sf.write('ac.wav', ac * sm, args.sr, subtype='PCM_16')

    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(411)
    plt.plot(ac)
    plt.subplot(412)
    plt.plot(dc * sm)
    plt.subplot(413)
    plt.plot(ac * sm)
    plt.subplot(414)
    plt.plot((dc+ac) * sm)
    plt.show()

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
    parser.add_argument('--sr', type=int, default=48000)
    parser.add_argument('--time', type=int, default=10.)
    #------------------------------ 
    args = parser.parse_args()

    prepare_pilot_signal(args)


