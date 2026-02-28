import torch
import os
import random
import numpy as np

def is_dist_avail_and_initialized():
    """Kiểm tra xem hệ thống distributed có sẵn sàng không."""
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True

def get_rank():
    """Lấy rank hiện tại, trả về 0 nếu không chạy distributed."""
    if not is_dist_avail_and_initialized():
        return 0
    return torch.distributed.get_rank()

def initialize_distributed(args):
    """
    Khởi tạo distributed. 
    Nếu không tìm thấy biến môi trường, hàm sẽ bỏ qua (dành cho Single GPU).
    """
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print("| Single GPU mode detected. Skipping distributed init.")
        args.rank = 0
        args.world_size = 1
        args.gpu = 0
        args.distributed = False
        return

    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.gpu = int(os.environ['LOCAL_RANK'])
    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    
    torch.distributed.init_process_group(
        backend=args.dist_backend, 
        init_method=args.dist_url,
        world_size=args.world_size, 
        rank=args.rank
    )
    torch.distributed.barrier()

def set_random_seed(seed):
    """Thiết lập seed để đảm bảo kết quả có thể tái lập."""
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # Thêm cho cả CUDA

def mkdir_ckpt_dirs(args):
    """Tạo thư mục lưu trữ, chỉ thực hiện trên Rank 0."""
    if get_rank() == 0:
        if os.path.exists(args.save):
            # Thay vì exit(), ta chỉ in cảnh báo để tránh dừng chương trình khi debug
            print(f'Warning: savedir already exists at {args.save}')
        
        os.makedirs(args.save, exist_ok=True)
        os.makedirs(os.path.join(args.save, 'ckpts'), exist_ok=True)
        os.makedirs(os.path.join(args.save, 'samples'), exist_ok=True)
        
        # Lưu lại cấu hình (setting.txt)
        args_dict = args.__dict__
        with open(os.path.join(args.save, 'setting.txt'), 'w') as f:
            f.writelines('------------------- start -------------------' + '\n')
            for arg, value in args_dict.items():
                f.writelines(f'{arg} : {value}\n')
            f.writelines('------------------- end -------------------' + '\n')

def multiply_list(my_list):
    """Nhân các phần tử trong danh sách."""
    result = 1
    for x in my_list:
        result = result * x
    return result