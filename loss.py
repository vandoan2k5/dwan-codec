import torch
import torch.nn.functional as F

class MultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(self, resolutions=[(512, 120, 512), (1024, 240, 1024), (2048, 480, 2048)]):
        super().__init__()
        self.resolutions = resolutions
        
        # Tạo sẵn các window và đăng ký dưới dạng buffer để không phải tạo lại mỗi vòng lặp
        for n_fft, hop_length, win_length in self.resolutions:
            self.register_buffer(f'window_{win_length}', torch.hann_window(win_length))

    def forward(self, x, y):
        # x, y shape: (B, 1, T)
        loss = 0.0
        x = x.squeeze(1)
        y = y.squeeze(1)
        
        for n_fft, hop_length, win_length in self.resolutions:
            # Lấy window đã được tạo sẵn trên GPU
            window = getattr(self, f'window_{win_length}')
            
            x_stft = torch.stft(x, n_fft, hop_length, win_length, 
                                window=window, 
                                return_complex=True)
            y_stft = torch.stft(y, n_fft, hop_length, win_length, 
                                window=window, 
                                return_complex=True)
            
            # Thêm eps 1e-7 để tránh log(0) và đạo hàm bị NaN
            x_mag = torch.abs(x_stft) + 1e-7
            y_mag = torch.abs(y_stft) + 1e-7
            
            # Spectral convergence loss (Chuẩn Frobenius)
            sc_loss = torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
            
            # Log STFT magnitude loss (Chuẩn L1)
            log_loss = F.l1_loss(torch.log(y_mag), torch.log(x_mag))
            
            loss += sc_loss + log_loss
            
        return loss / len(self.resolutions)

# --- Thêm vào loss.py ---

# 1. GAN Loss dành cho Discriminator (LSGAN style)
def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    # Ép logits real về 1, logits fake về 0
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((dr - 1)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
    return loss / len(disc_real_outputs)

# 2. GAN Loss dành cho Generator (LSGAN style)
def generator_loss(disc_generated_outputs):
    loss = 0
    # Ép logits fake về 1 (Generator muốn "lừa" D)
    for dg in disc_generated_outputs:
        l = torch.mean((dg - 1)**2)
        loss += l
    return loss / len(disc_generated_outputs)

# 3. Feature Matching Loss (Cực quan trọng để ổn định training và khử tiếng rè)
def feature_matching_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for cr, cg in zip(dr, dg):
            # Tính L1 giữa feature maps của từng tầng
            loss += torch.mean(torch.abs(cr - cg))
    # Chúng ta nhân với một hệ số nhỏ (ví dụ 2.0 hoặc 5.0) trong main.py
    return loss