import torch
import torchaudio
import numpy as np
import os
from argparse import Namespace
from .vae import VQVAE  

class Lumina:
    def __init__(self, config: Namespace, checkpoint_path: str, sample_rate: int, device='cuda'):
        """
        Wrapper class cho Lumina VQ-VAE.
        
        Args:
            config: Namespace chứa tham số model (in_channel, channel, etc.)
            checkpoint_path: Đường dẫn file .pt/.pth
            device: 'cuda' hoặc 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.target_sr = sample_rate  # Đặt cứng hoặc lấy từ config nếu có
        
        # 1. Khởi tạo Model gốc
        self.model = VQVAE(config)
        self.model.to(self.device)
        self.model.eval()

        # 2. Load Checkpoint (Xử lý lỗi key thông minh)
        self._load_checkpoint(checkpoint_path)

        # Tính toán factor downsample để pad audio (8*8*8 = 512)
        # Nếu cấu trúc thay đổi, cần sửa số này
        self.downsample_factor = 512 

    def _load_checkpoint(self, path):
        print(f"Loading checkpoint from: {path}")
        ckpt = torch.load(path, map_location='cpu')
        
        # Unwrap state_dict nếu nó bị lồng trong dict khác (nguyên nhân lỗi của bạn)
        if 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
            
        # Xử lý trường hợp train bằng DataParallel (có tiền tố 'module.')
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v # Bỏ 'module.'
            else:
                new_state_dict[k] = v
                
        # Load strict=False để tránh crash nếu thiếu vài key không quan trọng (optional)
        try:
            self.model.load_state_dict(new_state_dict, strict=True)
            print("Successfully loaded state dict!")
        except RuntimeError as e:
            print(f"Warning during loading (check keys carefully): {e}")
            # Thử load lại với strict=False nếu bạn chấp nhận rủi ro
            self.model.load_state_dict(new_state_dict, strict=False)

    def _preprocess_audio(self, audio_path_or_tensor):
        """Xử lý audio: Load -> Mono -> Resample -> Pad"""
        if isinstance(audio_path_or_tensor, str):
            wav, sr = torchaudio.load(audio_path_or_tensor)
        else:
            wav = audio_path_or_tensor
            sr = self.target_sr # Giả định đúng SR nếu đưa tensor vào

        # Resample
        if sr != self.target_sr:
            transform = torchaudio.transforms.Resample(sr, self.target_sr)
            wav = transform(wav)

        # Mix to Mono
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)

        # Pad cho chia hết stride
        _, length = wav.shape
        remainder = length % self.downsample_factor
        if remainder != 0:
            pad_len = self.downsample_factor - remainder
            wav = torch.nn.functional.pad(wav, (0, pad_len))
            
        return wav.unsqueeze(0).to(self.device) # (1, 1, T)

    @torch.no_grad()
    def encode(self, audio_input):
        """
        Mã hóa audio thành codes.
        
        Args:
            audio_input: Đường dẫn file wav HOẶC torch tensor.
            
        Returns:
            codes (numpy.ndarray): Mảng 1D chứa các indices.
        """
        x = self._preprocess_audio(audio_input)
        
        # model.encode trả về: quant, diff, indices
        # indices shape: (B, L)
        _, _, indices = self.model.encode(x)
        
        # Flatten về 1D array để dễ lưu/xử lý với LM
        return indices.squeeze().cpu().numpy()

    @torch.no_grad()
    def decode(self, codes):
        """
        Giải mã codes thành audio.
        
        Args:
            codes: Numpy array (1D hoặc 2D) hoặc torch tensor chứa indices.
            
        Returns:
            wavs (torch.Tensor): Waveform (1, T) trên CPU.
            sr (int): Sample rate.
        """
        # Chuẩn hóa input về tensor (B, L)
        if isinstance(codes, np.ndarray):
            codes = torch.from_numpy(codes).to(self.device).long()
        elif isinstance(codes, torch.Tensor):
            codes = codes.to(self.device).long()
            
        if codes.dim() == 1:
            codes = codes.unsqueeze(0) # Thêm batch dim -> (1, L)
            
        # Decode từ indices
        # Hàm decode_code: indices -> quant -> decoder -> wav
        wav_tensor = self.model.decode_code(codes)
        
        # Output shape của model thường là (B, 1, T) hoặc (B, T)
        # Ta đưa về (1, T) trên CPU
        wav_tensor = wav_tensor.squeeze(0).cpu() 
        
        return wav_tensor, self.target_sr