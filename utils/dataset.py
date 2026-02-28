import torch
import torchaudio
import os
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, data_path, sample_rate=24000, duration=1.0):
        self.data_path = data_path
        self.sample_rate = sample_rate
        self.num_samples = int(sample_rate * duration)
        
        # Kiểm tra đường dẫn tồn tại
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data path not found: {data_path}")

        # Lấy danh sách các tệp âm thanh
        self.file_list = [os.path.join(data_path, f) for f in os.listdir(data_path) 
                          if f.endswith(('.wav', '.flac', '.mp3'))]
        
        if len(self.file_list) == 0:
            print(f"Warning: No audio files found in {data_path}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Nạp tệp âm thanh
        waveform, sr = torchaudio.load(self.file_list[idx])
        
        # Chuyển về Mono nếu là Stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            
        # Resample nếu tần số lấy mẫu không khớp
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
            
        # Đảm bảo độ dài cố định (Cắt hoặc Pad)
        if waveform.shape[1] > self.num_samples:
            waveform = waveform[:, :self.num_samples]
        else:
            padding = self.num_samples - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
            
        return waveform, 0

def get_data_loaders(args):
    """
    Khởi tạo DataLoader hỗ trợ cả chế độ Single-GPU và Distributed.
    """
    train_set = AudioDataset(args.train_data_path, sample_rate=args.sample_rate, duration=args.duration)
    val_set = AudioDataset(args.val_data_path, sample_rate=args.sample_rate, duration=args.duration)

    # Kiểm tra xem distributed có đang chạy không
    is_dist = torch.distributed.is_available() and torch.distributed.is_initialized()

    if is_dist:
        # Chế độ Distributed
        sampler_train = torch.utils.data.distributed.DistributedSampler(
            train_set, 
            num_replicas=torch.distributed.get_world_size(), 
            rank=torch.distributed.get_rank(), 
            shuffle=True
        )
        # Đối với Validation, thường không cần sampler trừ khi tập val cực lớn
        sampler_val = None 
        shuffle_train = False
    else:
        # Chế độ Single-GPU
        sampler_train = None
        sampler_val = None
        shuffle_train = True

    train_data_loader = torch.utils.data.DataLoader(
        train_set, 
        sampler=sampler_train,
        shuffle=shuffle_train, # Chỉ shuffle nếu không có sampler
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        pin_memory=True # Tăng tốc độ chuyển dữ liệu lên GPU
    )
    
    val_data_loader = torch.utils.data.DataLoader(
        val_set,
        sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )
    
    return train_data_loader, val_data_loader