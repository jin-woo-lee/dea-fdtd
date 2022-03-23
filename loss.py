import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as TAF
import constants as C

class SpecLoss(nn.Module):
    def __init__(
            self,
            n_fft, win_length=None, hop_length=None, window='hann',
            n_mels=80, normalize=True,
        ):
        super().__init__()
        sr = C.anal_sr
        self.n_fft = n_fft
        self.win_length = win_length if win_length else n_fft
        self.hop_length = hop_length if hop_length else n_fft // 4

        window = librosa.filters.get_window(window, win_length)
        self.window = torch.from_numpy(window).cuda()
        self.normalize = normalize

        basis = librosa.filters.mel(sr, win_length, n_mels=n_mels, fmin=0, fmax=sr//2)
        self.basis = torch.from_numpy(basis).float().cuda()

        self.w = nn.Parameter(torch.Tensor([[[1.]]]))

    def forward(self, est, gt):
    
        est_spec = TAF.spectrogram(
            est,
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            pad=self.win_length // 2, 
            window=self.window, 
            normalized=self.normalize, 
            power=1,    # 1: energy, 2: power, None: cplx
        ).float()
        gt_spec = TAF.spectrogram(
            gt,
            n_fft=self.n_fft, 
            hop_length=self.hop_length, 
            win_length=self.win_length, 
            pad=self.win_length // 2, 
            window=self.window, 
            normalized=self.normalize, 
            power=1,    # 1: energy, 2: power, None: cplx
        ).float()
        est_spec = torch.matmul(self.basis, est_spec)
        gt_spec  = torch.matmul(self.basis, gt_spec)

        est_spec = self.w * est_spec

        return F.mse_loss(est_spec, gt_spec)

