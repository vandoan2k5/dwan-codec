import os
import torch
import torchaudio
from arguments import get_args
from models import VQVAE
from discriminators import MultiDiscriminator
from dataset import get_data_loaders
from loss import MultiResolutionSTFTLoss, discriminator_loss, generator_loss, feature_matching_loss
from utils import set_random_seed, mkdir_ckpt_dirs
from scheduler import AnnealingLR
from tqdm import tqdm
import wandb

def load_checkpoint(ckpt_path, model_g, model_d, opt_g, opt_d, sch_g, sch_d, device):
    """Cơ chế load thông minh: Hỗ trợ load từ checkpoint VQ-VAE cũ (33k steps) hoặc checkpoint GAN mới."""
    print(f"🔄 Đang tải checkpoint từ: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 1. Nếu là checkpoint GAN mới (đã có model_g_state_dict)
    if 'model_g_state_dict' in checkpoint:
        model_g.load_state_dict(checkpoint['model_g_state_dict'])
        model_d.load_state_dict(checkpoint['model_d_state_dict'])
        opt_g.load_state_dict(checkpoint['opt_g_state_dict'])
        opt_d.load_state_dict(checkpoint['opt_d_state_dict'])
        sch_g.load_state_dict(checkpoint['sch_g_state_dict'])
        sch_d.load_state_dict(checkpoint['sch_d_state_dict'])
        epoch = checkpoint['epoch']
        num_iter = checkpoint['iter']
        print("✅ Tải thành công toàn bộ hệ thống GAN!")
        
    # 2. Nếu là checkpoint VQ-VAE cũ (mốc 33k steps)
    else:
        model_g.load_state_dict(checkpoint['model_state_dict'])
        opt_g.load_state_dict(checkpoint['optimizer_state_dict'])
        sch_g.load_state_dict(checkpoint['scheduler_state_dict'])
        epoch = checkpoint['epoch']
        num_iter = checkpoint['iter']
        print("⚠️ Tải thành công VQ-VAE cũ. Discriminator được khởi tạo mới 100%.")
        
    return epoch, num_iter

def main():
    args = get_args()
    wandb.init(
        project="codec-fsq", 
        config=vars(args),              
        name=f"run_GAN_{args.lr_decay_style}_{args.lr}" 
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"--- 🚀 Bắt đầu huấn luyện hệ thống Audio GAN trên: {device} ---")
    
    set_random_seed(args.seed)
    mkdir_ckpt_dirs(args)

    train_data_loader, val_data_loader = get_data_loaders(args)

    # Khởi tạo Generator (VQ-VAE) và Cảnh sát (Discriminator)
    model_g = VQVAE(args).to(device)
    model_d = MultiDiscriminator().to(device)
    
    # Khởi tạo Optimizer (Dùng AdamW với beta chuẩn cho Audio GAN)
    optim_g = torch.optim.AdamW(model_g.parameters(), lr=args.lr, betas=(0.8, 0.99), weight_decay=args.weight_decay)
    # Discriminator thường để LR nhỉnh hơn G một chút hoặc bằng G
    lr_d = getattr(args, 'lr_disc', args.lr) 
    optim_d = torch.optim.AdamW(model_d.parameters(), lr=lr_d, betas=(0.8, 0.99), weight_decay=args.weight_decay)
    
    # Schedulers
    sch_g = AnnealingLR(optim_g, start_lr=args.lr, warmup_iter=args.warmup * args.train_iters, num_iters=args.train_iters, decay_style=args.lr_decay_style, last_iter=-1, decay_ratio=args.lr_decay_ratio)
    sch_d = AnnealingLR(optim_d, start_lr=lr_d, warmup_iter=args.warmup * args.train_iters, num_iters=args.train_iters, decay_style=args.lr_decay_style, last_iter=-1, decay_ratio=args.lr_decay_ratio)

    # Load Resume (Nếu có đường dẫn checkpoint)
    start_epoch, num_iter = 0, 0
    resume_path = getattr(args, 'resume_path', None)
    if resume_path and os.path.exists(resume_path):
        start_epoch, num_iter = load_checkpoint(resume_path, model_g, model_d, optim_g, optim_d, sch_g, sch_d, device)

    # Khởi tạo Loss Functions (fp32)
    get_l1loss = torch.nn.L1Loss()
    stft_loss_fn = MultiResolutionSTFTLoss().to(device)

    print(f"✨ Hệ thống đã sẵn sàng. Bắt đầu từ Epoch {start_epoch}, Iter {num_iter}...")
    bar = tqdm(range(start_epoch, args.max_train_epochs), initial=start_epoch, total=args.max_train_epochs)
    
    for epoch in bar:
        model_g.train()
        model_d.train()
        
        for _, (input_audio, _) in enumerate(train_data_loader):
            num_iter += 1
            input_audio = input_audio.to(device)
            
            # -------------------------------------------------------------
            # [PHẦN 1: FORWARD VỚI BFLOAT16]
            # -------------------------------------------------------------
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                y_g_hat, codebook_loss, _ = model_g(input_audio)
                
            recon_len = y_g_hat.shape[2]
            target_audio = input_audio[:, :, :recon_len]
            
            # Chuyển về float32 để tính toán Loss và Discriminator ổn định
            y_g_hat_f32 = y_g_hat.float()
            target_audio_f32 = target_audio.float()

            # -------------------------------------------------------------
            # [PHẦN 2: HUẤN LUYỆN DISCRIMINATOR]
            # -------------------------------------------------------------
            optim_d.zero_grad()
            # Dùng y_g_hat_f32.detach() để ngắt gradient không truyền về Generator
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                y_d_r, y_d_g, _, _ = model_d(target_audio_f32, y_g_hat_f32.detach())
                
            loss_disc = discriminator_loss(y_d_r, y_d_g)
            loss_disc.backward()
            optim_d.step()
            sch_d.step()
            
            # -------------------------------------------------------------
            # [PHẦN 3: HUẤN LUYỆN GENERATOR (VQ-VAE)]
            # -------------------------------------------------------------
            optim_g.zero_grad()
            
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                y_d_r, y_d_g, fmap_r, fmap_g = model_d(target_audio_f32, y_g_hat_f32)
                
            loss_gan_g = generator_loss(y_d_g)
            loss_fm = feature_matching_loss(fmap_r, fmap_g)
            
            l1_loss_val = get_l1loss(y_g_hat_f32, target_audio_f32)
            stft_loss_val = stft_loss_fn(y_g_hat_f32, target_audio_f32)
            
            # Tính toán Total Loss cho Generator
            loss_gen_total = (args.l1_weight * l1_loss_val) + \
                             (args.stft_weight * stft_loss_val) + \
                             (args.codebook_weight * codebook_loss) + \
                             (getattr(args, 'gan_weight', 1.0) * loss_gan_g) + \
                             (getattr(args, 'fm_weight', 2.0) * loss_fm)
            
            loss_gen_total.backward()
            optim_g.step()
            sch_g.step()
            
            # -------------------------------------------------------------
            # [PHẦN 4: LOGGING & SAVING]
            # -------------------------------------------------------------
            if num_iter % 5 == 0:
                bar.set_description(f"Epoch:[{epoch}] | Iter:{num_iter} | "
                      f"D_Loss:{loss_disc.item():.3f} | G_GAN:{loss_gan_g.item():.3f} | "
                      f"STFT:{stft_loss_val.item():.3f} | FM:{loss_fm.item():.3f}")

                wandb.log({
                    "train_Generator/total_loss": loss_gen_total.item(),
                    "train_Generator/stft_loss": stft_loss_val.item(),
                    "train_Generator/l1_loss": l1_loss_val.item(),
                    "train_Generator/codebook_loss": codebook_loss.item(),
                    "train_GAN/generator_adversarial_loss": loss_gan_g.item(),
                    "train_GAN/feature_matching_loss": loss_fm.item(),
                    "train_Discriminator/discriminator_loss": loss_disc.item(),
                    "learning_rate/generator": optim_g.param_groups[0]['lr'],
                    "learning_rate/discriminator": optim_d.param_groups[0]['lr'],
                    "epoch": epoch
                }, step=num_iter)
            
            if num_iter % args.log_interval == 0:
                save_samples(args, num_iter, target_audio_f32, y_g_hat_f32)

        if epoch % 5 == 0 or epoch == args.max_train_epochs - 1:
            save_checkpoint(args, epoch, num_iter, model_g, model_d, optim_g, optim_d, sch_g, sch_d)

def save_samples(args, num_iter, input_audio, reconstructions):
    output_dir = os.path.join(args.save, 'samples')
    os.makedirs(output_dir, exist_ok=True)
    
    sample_input = input_audio[0].detach().cpu()
    sample_recon = reconstructions[0].detach().cpu()
    
    torchaudio.save(os.path.join(output_dir, f'iter_{num_iter}_orig.wav'), sample_input, args.sample_rate)
    torchaudio.save(os.path.join(output_dir, f'iter_{num_iter}_recon.wav'), sample_recon, args.sample_rate)

def save_checkpoint(args, epoch, num_iter, model_g, model_d, opt_g, opt_d, sch_g, sch_d):
    ckpt_path = os.path.join(args.save, 'ckpts', f'epoch_{epoch}.pt')
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'iter': num_iter,
        'model_g_state_dict': model_g.state_dict(),
        'model_d_state_dict': model_d.state_dict(),
        'opt_g_state_dict': opt_g.state_dict(),
        'opt_d_state_dict': opt_d.state_dict(),
        'sch_g_state_dict': sch_g.state_dict(),
        'sch_d_state_dict': sch_d.state_dict()
    }, ckpt_path)
    print(f"💾 Đã lưu checkpoint GAN tại: {ckpt_path}")

if __name__ == "__main__":
    main()