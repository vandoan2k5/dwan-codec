import torch
from torch import nn
from einops import rearrange
from .encoder import Encoder
from .decoder import Decoder
from quantizers import VectorQuantizeEMA, FSQ, LFQ, FSQSTE

class VQVAE(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Khởi tạo Encoder
        self.enc = Encoder(
            in_channel=args.in_channel, 
            dims=getattr(args, 'encoder_dims', [128, 256, 512, 512]),
            depths=getattr(args, 'encoder_depths', [3, 3, 9, 3]),
            strides=getattr(args, 'strides', [8, 8, 8]),
            embed_dim=args.embed_dim
        )

        # Khởi tạo Quantizer
        if args.quantizer == 'ema':
            self.quantize_t = VectorQuantizeEMA(args, args.embed_dim, args.n_embed)
        elif args.quantizer == 'lfq':
            lfq_dim = getattr(args, 'lfq_dim', 10) 
            self.quantize_t = LFQ(codebook_size=2**lfq_dim, dim=lfq_dim)
        elif args.quantizer == 'fsq':
            self.quantize_t = FSQ(levels=args.levels)
        elif args.quantizer == 'fsqste':
            self.quantize_t = FSQSTE(levels=args.levels)
        else:
            raise ValueError(f"Unknown quantizer: {args.quantizer}")

        # Khởi tạo Decoder với cơ chế bảo vệ AttributeError
        self.dec = Decoder(
            embed_dim=args.embed_dim,
            decoder_dim=getattr(args, 'decoder_dim', 768),
            decoder_num_layers=getattr(args, 'decoder_num_layers', 8),
            upscale=getattr(args, 'upscale', 4),
            n_fft=getattr(args, 'n_fft', 2048),
            hop_length=getattr(args, 'hop_length', 512),
            dw_kernel=getattr(args, 'dw_kernel', 3)
        )

    def encode(self, x):
        z = self.enc(x)
        z = rearrange(z, 'b c l -> b l c')
        
        if self.args.quantizer == 'fsq':
            quant, indices = self.quantize_t(z)
            diff = torch.tensor(0.0).to(z.device)
        else:
            quant, diff, indices = self.quantize_t(z)
            
        quant = rearrange(quant, 'b l c -> b c l')
        return quant, diff, indices

    def decode(self, quant):
        return self.dec(quant)

    def forward(self, x, return_id=True):
        quant, diff, indices = self.encode(x)
        recon = self.decode(quant)
        if return_id:
            return recon, diff, indices
        return recon, diff

    def decode_code(self, indices):
        quant = self.quantize_t.embed_code(indices)
        quant = rearrange(quant, 'b l c -> b c l')
        return self.dec(quant)