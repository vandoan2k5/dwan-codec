import torch
import torchaudio
import argparse
import os
from utils.arguments import add_model_config_args, add_data_args, get_args
from models import VQVAE
from torchinfo import summary

def preprocess_audio(wav, sr, target_sr, hop_length):
    """
    Xử lý audio đầu vào: Resample -> Mono -> Pad
    """
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        wav = resampler(wav)
    
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
        
    length = wav.shape[-1]
    remainder = length % hop_length
    if remainder != 0:
        pad_len = hop_length - remainder
        wav = torch.nn.functional.pad(wav, (0, pad_len))
        
    return wav

def get_inference_args():
    """Khởi tạo Parser dành riêng cho lúc Inference (Test)"""
    parser = argparse.ArgumentParser(description="Chạy thử mô hình Lumina VQ-VAE")
    
    parser = add_model_config_args(parser)
    parser = add_data_args(parser)
    
    parser.add_argument('--ckpt', type=str, required=True, help='Đường dẫn tới file checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Đường dẫn file audio gốc (.wav)')
    parser.add_argument('--output', type=str, default='sample.wav', help='Đường dẫn lưu file kết quả')
    
    args = parser.parse_args()
    
    if args.quantizer == 'fsq':
        args.embed_dim = len(args.levels)
        
    return args

def main():
    args = get_inference_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Đang chạy inference trên: {device}")

    model = VQVAE(args)
    batch_size = 8
    dummy_input = torch.randn(batch_size, 1, 24000)

    summary(model, input_data=dummy_input, 
            col_names=["input_size", "output_size", "num_params", "trainable"],
            col_width=20,
            row_settings=["var_names"])
    model = model.to(device)
    print(f"📦 Đang load weights từ: {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location=device)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
        
    clean_state_dict = {}
    for k, v in state_dict.items():
        clean_key = k.replace('module.', '') if k.startswith('module.') else k
        clean_state_dict[clean_key] = v
    
    model.load_state_dict(clean_state_dict, strict=True)
    model.eval()

    print(f"🎵 Đang xử lý file audio: {args.input}")
    wav, sr = torchaudio.load(args.input)
    
    hop_length = args.strides[0] * args.strides[1] * args.strides[2] if hasattr(args, 'strides') else 512
    
    wav = preprocess_audio(wav, sr, args.sample_rate, hop_length)
    wav = wav.unsqueeze(0).to(device) # (1, 1, T)

    print("🧠 Mô hình đang lượng tử hóa và tái tạo âm thanh...")
    with torch.no_grad():
        recon_audio, _, indices = model(wav)
        recon_audio = recon_audio.squeeze(0).cpu()

    torchaudio.save(args.output, recon_audio, args.sample_rate)
    
    print(f"✅ Hoàn tất! File tái tạo đã được lưu tại: {args.output}")
    print(f"📊 Thông tin Codebook: Độ dài chuỗi token mã hóa là {indices.shape[1]}")

if __name__ == "__main__":
    main()