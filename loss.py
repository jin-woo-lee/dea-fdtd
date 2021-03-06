import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as TAT
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

        self.to_mel = TAT.MelSpectrogram(
            sample_rate = C.anal_sr,
            n_fft=n_fft, win_length=win_length, hop_length=hop_length,
            f_min=0, f_max=sr//2, n_mels=n_mels, normalized=normalize,
        )
        self.coeff_w = nn.Parameter(torch.Tensor([[1.]]))
        #self.coeff_b = nn.Parameter(torch.Tensor([[0.]]))

    def forward(self, est, gt, guide=False):
    
        est = self.coeff_w * est
        #est = self.coeff_b + est

        est_spec = self.to_mel(est)
        gt_spec  = self.to_mel(gt)

        loss = F.mse_loss(est_spec, gt_spec)
        if guide:
            loss += F.mse_loss(est, gt)

        return loss

