import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm, spectral_norm

# helper functions
def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)

# --- 1. Multi-Period Discriminator (MPD) ---
class DiscriminatorP(nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super().__init__()
        self.period = period
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        
        # Conv2D vì chúng ta reshape audio thành 2D (T/Period, Period)
        self.convs = nn.ModuleList([
            norm_f(nn.Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(nn.Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(nn.Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = [] # Lưu feature maps để tính Feature Matching Loss

        # 1D audio (B, 1, T) -> Reshape to 2D dựa trên period
        b, c, t = x.shape
        if t % self.period != 0: # Pad cho chia hết
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1) # Logits

        return x, fmap

class MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # Thử nghiệm ở các chu kỳ nguyên tố để bao phủ mọi khoảng cách pha
        self.discriminators = nn.ModuleList([
            DiscriminatorP(2),
            DiscriminatorP(3),
            DiscriminatorP(5),
            DiscriminatorP(7),
            DiscriminatorP(11),
        ])

    def forward(self, y, y_hat):
        y_d_rs = [] # Logits của audio gốc (Real)
        y_d_gs = [] # Logits của audio sinh ra (Fake)
        fmap_rs = [] # Feature maps của audio gốc
        fmap_gs = [] # Feature maps của audio sinh ra
        
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

# --- 2. Multi-Scale Discriminator (MSD) ---
class DiscriminatorS(nn.Module):
    def __init__(self, use_spectral_norm=False):
        super().__init__()
        norm_f = weight_norm if not use_spectral_norm else spectral_norm
        
        # 1D Conv thông thường trên nhiều độ phân giải
        self.convs = nn.ModuleList([
            norm_f(nn.Conv1d(1, 128, 15, 1, padding=7)),
            norm_f(nn.Conv1d(128, 128, 41, 4, groups=4, padding=20)),
            norm_f(nn.Conv1d(128, 256, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(256, 512, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(512, 1024, 41, 4, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 41, 1, groups=16, padding=20)),
            norm_f(nn.Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(nn.Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap

class MultiScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        # 3 mức độ phân giải: Nguyên bản, downsample x2, downsample x4
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=True), # Spectral Norm cho bản gốc để ổn định
            DiscriminatorS(),
            DiscriminatorS(),
        ])
        self.meanpools = nn.ModuleList([
            nn.AvgPool1d(4, 2, padding=2),
            nn.AvgPool1d(4, 2, padding=2),
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        
        # Xử lý độ phân giải cao nhất trước
        y_d_r, fmap_r = self.discriminators[0](y)
        y_d_g, fmap_g = self.discriminators[0](y_hat)
        y_d_rs.append(y_d_r)
        fmap_rs.append(fmap_r)
        y_d_gs.append(y_d_g)
        fmap_gs.append(fmap_g)
        
        # Downsample và xử lý các độ phân giải thấp hơn
        for i, d in enumerate(self.discriminators[1:]):
            y = self.meanpools[i](y)
            y_hat = self.meanpools[i](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs

# --- Wrapper cuối cùng cho Model VQ-VAE gọi ---
class MultiDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mpd = MultiPeriodDiscriminator()
        self.msd = MultiScaleDiscriminator()

    def forward(self, y, y_hat):
        # Gom kết quả từ cả MPD và MSD
        logits_real_mpd, logits_fake_mpd, fmaps_real_mpd, fmaps_fake_mpd = self.mpd(y, y_hat)
        logits_real_msd, logits_fake_msd, fmaps_real_msd, fmaps_fake_msd = self.msd(y, y_hat)
        
        # Nối list kết quả lại
        logits_real = logits_real_mpd + logits_real_msd
        logits_fake = logits_fake_mpd + logits_fake_msd
        fmaps_real = fmaps_real_mpd + fmaps_real_msd
        fmaps_fake = fmaps_fake_mpd + fmaps_fake_msd
        
        return logits_real, logits_fake, fmaps_real, fmaps_fake