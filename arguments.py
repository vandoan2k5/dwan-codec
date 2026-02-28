"""Cấu hình bộ nạp tham số (argparser) cho Audio"""

import argparse
import os
import torch
from utils import multiply_list

def add_model_config_args(parser):
    """Tham số cấu hình mô hình"""
    group = parser.add_argument_group('model', 'model configuration')

    # Encoder Audio
    group.add_argument('--in-channel', type=int, default=1, help="Số kênh đầu vào (1 cho Mono, 2 cho Stereo)")
    group.add_argument('--channel', type=int, default=512, help="Số kênh trung gian trong mạng Conv1D")
    group.add_argument('--encoder-dims', nargs='+', type=int, default=[128, 256, 512, 512], help="Số channel từng block của Encoder")
    group.add_argument('--encoder-depths', nargs='+', type=int, default=[3, 3, 9, 3], help="Số lượng ConvNeXt block mỗi stage")
    group.add_argument('--strides', nargs='+', type=int, default=[8, 8, 8], help="Hệ số downsample (Tích của list này bằng hop_length)")
    
    # Decoder Audio (Các tham số bị thiếu)
    group.add_argument('--decoder-dim', type=int, default=768, help="Kích thước ẩn của Decoder Vocos")
    group.add_argument('--decoder-num-layers', type=int, default=8, help="Số lớp ConvNeXt trong Decoder")
    group.add_argument('--dw-kernel', type=int, default=3, help="Kernel size cho Depthwise Conv trong Decoder")
    group.add_argument('--upscale', type=int, default=4, help="Hệ số upscale (nếu có dùng F.interpolate)")
    
    # ISTFT Head
    group.add_argument('--n-fft', type=int, default=1024, help="Kích thước cửa sổ FFT cho ISTFT")
    group.add_argument('--hop-length', type=int, default=512, help="Bước nhảy khung hình (Hop size)")
    
    # VQ-VAE Quantizer
    group.add_argument('--quantizer', type=str, default="fsq", choices=['ema','origin','fsq','lfq', 'fsqste'])
    group.add_argument('--levels', nargs='+', type=int, default=[8, 8, 8, 6, 5], help='Các mức chia cho FSQ (Mặc định 16k codes)')
    group.add_argument('--lfq-dim', type=int, default=14, help='Kích thước cho LFQ (nếu dùng)')
    group.add_argument('--embed-dim', type=int, default=768)
    group.add_argument('--n-embed', type=int, default=5)
    return parser

def add_training_args(parser):
    """Tham số huấn luyện"""
    group = parser.add_argument_group('train', 'training configurations')

    group.add_argument('--experiment-name', type=str, default="Audio_FSQ",
                       help="Tên thí nghiệm để lưu log và checkpoint")
    group.add_argument('--batch-size', type=int, default=128,
                       help='Số lượng mẫu trong một batch')
    
    # BỔ SUNG CÁC THAM SỐ CÒN THIẾU
    group.add_argument('--weight-decay', type=float, default=0.,
                       help='Hệ số weight decay cho L2 regularization')
    group.add_argument('--lr-decay-style', type=str, default='linear',
                       choices=['constant', 'linear', 'cosine', 'exponential'],
                       help='Kiểu giảm tốc độ học')
    group.add_argument('--lr-decay-ratio', type=float, default=0.1)
    group.add_argument('--warmup', type=float, default=0.01,
                       help='Phần trăm số iter để warmup tốc độ học')
    
    group.add_argument('--train-iters', type=int, default=500000)
    group.add_argument('--max-train-epochs', type=int, default=200)
    group.add_argument('--log-interval', type=int, default=1000)
    group.add_argument('--seed', type=int, default=1234)
    group.add_argument('--lr', type=float, default=1.0e-4)
    group.add_argument('--save', type=str, default=None,
                       help='Thư mục lưu kết quả.')
    
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')
    group.add_argument('--lr-disc', type=float, default=2e-4, 
                    help='Tốc độ học của Discriminator (Thường nhỉnh hơn Generator một chút)')
    
    return parser

def add_data_args(parser):
    """Tham số cấu hình dữ liệu âm thanh"""
    group = parser.add_argument_group('data', 'data configurations')
    
    # Đường dẫn dữ liệu Audio trên Kaggle
    group.add_argument('--train-data-path', type=str, default="/kaggle/codec/vivoice_250k_samples/train",
                        help="Đường dẫn thư mục chứa file .wav train")
    group.add_argument('--val-data-path', type=str, default="/kaggle/codec/vivoice_250k_samples/val", 
                       help="Đường dẫn thư mục chứa file .wav val")
    
    # Các thông số đặc thù cho Audio (Đã dùng trong dataset.py ở Bước 1)
    group.add_argument('--sample-rate', type=int, default=24000,
                        help='Tần số lấy mẫu âm thanh (Hz)')
    group.add_argument('--duration', type=float, default=1.0,
                        help='Độ dài đoạn âm thanh tính bằng giây')
    
    group.add_argument('--num-workers', type=int, default=2, # Kaggle nên để 2 hoặc 4
                       help="Số lượng worker nạp dữ liệu")
    return parser

def add_loss_args(parser):
    """Tham số trọng số các hàm mất mát"""
    group = parser.add_argument_group('weight', 'loss configurations')
   
    group.add_argument('--l1-weight', type=float, default=1.0, help='Trọng số của L1 Waveform loss.')
    group.add_argument('--stft-weight', type=float, default=10.0, help='Trọng số của Multi-Resolution STFT loss.')
    group.add_argument('--codebook-weight', type=float, default=1.0, help='Trọng số của quantization loss.')
    group.add_argument('--perceptual-weight', type=float, default=0., help='Trọng số perceptual (nếu có).')

    # Bổ sung vào hàm add_loss_args(parser):
    group.add_argument('--gan-weight', type=float, default=1.0, 
                    help='Trọng số của Adversarial GAN Loss')
    group.add_argument('--fm-weight', type=float, default=2.0, 
                    help='Trọng số của Feature Matching Loss (Khử rè cực tốt)')
    return parser

def get_args():
    """Phân tích toàn bộ tham số"""
    parser = argparse.ArgumentParser(description='PyTorch Audio FSQ-VQVAE')
    parser = add_model_config_args(parser)
    parser = add_training_args(parser)
    parser = add_data_args(parser)
    parser = add_loss_args(parser)
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    args.rank = int(os.getenv('RANK', '0'))
    args.world_size = int(os.getenv('WORLD_SIZE', '1'))
    
    # Tự động tạo tên thư mục lưu dựa trên quantizer
    if args.save is None:
        args.save = '.checkpoints/audio_' + str(args.quantizer) 

    if args.quantizer == 'fsq':
        args.embed_dim = len(args.levels)
        args.save += '-n_embed-' + str(multiply_list(args.levels))
    else:
        args.save += '-n_embed-' + str(args.n_embed)
        
    return args