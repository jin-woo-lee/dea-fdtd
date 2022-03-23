import numpy as np
import librosa
import scipy.signal
import matplotlib.pyplot as plt

import constants as C

def renormalize(x, y, eps=1e-12):
    mu = y.mean()
    std = y.std()
    x = x - x.mean()
    x /= np.clip(x.std(), a_min=eps, a_max=np.inf)
    x *= std
    return x + mu

def plot(t, i, freq, lam, vol, n_fft=1024, t_len=2048, do_filter=True):
    t = np.linspace(0,C.t_sup,int(C.t_sup*C.save_sr))
    f = np.linspace(0, C.save_sr//2, 1+n_fft//2)
    if C.anal_sr > C.save_sr:
        n_p = 100
        lam = np.pad(lam, (n_p*8,n_p*8), 'edge')
        vol = np.pad(vol, (n_p*8,n_p*8), 'edge')
        lam = librosa.resample(lam, C.anal_sr, C.save_sr)[n_p:-n_p]
        vol = librosa.resample(vol, C.anal_sr, C.save_sr)[n_p:-n_p]
    if C.do_filter:
        sos = scipy.signal.butter(10, 20, 'hp', fs=C.save_sr, output='sos')
        x = scipy.signal.sosfilt(sos, lam-1) + 1
    else:
        x = lam
    _x = x[- t_len:] - np.mean(x[- t_len:])
    _x = _x / (np.std(_x) + 1e-5)
    tf = np.fft.rfft(_x,n_fft)
    dB = np.log(np.absolute(tf) + 1e-5 )

    eps = 1e-4
    os.makedirs(f'plot', exist_ok=True)
    plt.figure(figsize=(7,7))
    plt.subplot(411)
    plt.title(f"Excitation Frequency {freq} Hz")
    plt.plot(t, vol, 'k-', label='stretch',linewidth=.7)
    plt.ylabel(f"$\Phi$")
    plt.subplot(412)
    plt.plot(t, lam, 'k-', label='stretch',linewidth=.7)
    #plt.plot(t[- t_len:], lam[- t_len:], 'k-', label='stretch',linewidth=.7)
    plt.ylabel(f"$\lambda$")
    plt.subplot(413)
    plt.plot(t, x, 'k-', label='filtered',linewidth=.7)
    #plt.plot(t[- t_len:], _x, 'k-', label='filtered',linewidth=.7)
    plt.xlabel(f"Time (sec)")
    plt.ylabel(f"$\lambda$")
    #plt.ylim([1-eps,1+eps])
    plt.tight_layout()
    plt.subplot(414)
    plt.plot(f, dB, 'k-', label='filtered',linewidth=.7)
    #plt.legend(bbox_to_anchor=(1.01, 1), loc=2, borderaxespad=0.)
    plt.title(f"frequency response")
    plt.xlabel(f"Frequency (Hz)")
    plt.ylabel(f"Amplitude (dB)")
    plt.tight_layout()
    plt.savefig(f'plot/{i}.png')
    plt.close()
    return np.expand_dims(x, -1)

def get_sine(freqs):
    """
    args
        freqs: array of frequencies for each batch (batch_size,)
    return
        array of sine with specified frequencies (batch_size, C.t_sup*C.anal_sr)
    """
    t = np.linspace(0,C.t_sup,int(C.t_sup*C.anal_sr))
    f_rel = np.tile(np.expand_dims(freqs,1), int(C.t_sup*C.anal_sr)) / C.anal_sr
    omega = np.cumsum(2*np.pi*f_rel, axis=1)
    return np.sin(omega)

def sample_sine(n, f=None):
    if f is None:
        #freq = np.random.randint(0,C.anal_sr // 2, size=n)
        freq = np.random.randint(0,C.anal_sr // 4, size=n)
        wav = np.random.uniform(C.Vpp,C.Vdc) + np.random.uniform(0,C.Vpp) * get_sine(freq) / 2
    else:
        freq = [f] * n
        wav = C.Vdc + C.Vpp * get_sine(freq)
    return wav

def sample_batch(n, f=None, num_timestep=None):
    #freq = np.random.randint(0,C.anal_sr // 2, size=n)
    freq = np.random.randint(0,C.anal_sr // 4, size=n)
    wav = np.random.uniform(C.Vpp,C.Vdc) + np.random.uniform(0,C.Vpp) * get_sine(freq) / 2
    if num_timestep:
        t = np.random.randint(0, wav.shape[-1]-num_timestep)
        wav = wav[:,t:t+num_timestep]
    return wav

def minmax_normalize(x, y=None):
    x_min = x.reshape(x.size(0),-1).min(-1).values.reshape(x.size(0),1)
    x = x - x_min
    x_max = x.reshape(x.size(0),-1).max(-1).values.reshape(x.size(0),1)
    x = x / x_max
    if y is not None:
        y_min = y.reshape(y.size(0),-1).min(-1).values.reshape(y.size(0),1)
        y_max = y.reshape(y.size(0),-1).max(-1).values.reshape(y.size(0),1)
        x = x * (y_max - y_min)
        x = x + y_min
    return x

def stretch_to_capacitance(lam):
    """
    C = eps_0 * eps_r * (A / d)
    """
    A = C.a_0 / lam
    d = C.z_0 * lam
    return C.eps * A / d


def cal_rms(amp):
    return np.sqrt(np.mean(np.square(amp), axis=-1))

def rms_normalize(wav, ref_dB=-23.0):
    # RMS normalize
    eps = np.finfo(np.float32).eps
    rms = cal_rms(wav)
    rms_dB = 20*np.log(rms/1) # rms_dB
    ref_linear = np.power(10, ref_dB/20.)
    gain = ref_linear / np.sqrt(np.mean(np.square(wav), axis=-1) + eps)
    wav = gain * wav
    return wav


