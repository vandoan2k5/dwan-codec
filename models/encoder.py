import torch
from torch import nn

class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block adapted from https://github.com/facebookresearch/ConvNeXt to 1D audio signal.

    Args:
        dim (int): Number of input channels.
        intermediate_dim (int): Dimensionality of the intermediate layer.
        layer_scale_init_value (float, optional): Initial value for the layer scale. None means no scaling.
            Defaults to None.
    """

    def __init__(
        self,
        dim: int,
        intermediate_dim: int,
        layer_scale_init_value: float,
        dw_kernel_size: int = 9,
    ):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=dw_kernel_size, padding=dw_kernel_size//2, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = (
            nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
            if layer_scale_init_value > 0
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        x = residual + x
        return x

class LuminaEncoder(nn.Module):
    def __init__(
        self, 
        in_channel=1, 
        dims=[128, 256, 512, 512], # Tăng dần channel: Nhẹ ở đầu, nặng ở đuôi
        depths=[3, 3, 9, 3],       # Số lượng block mỗi stage (QUAN TRỌNG: Sâu hơn nhiều)
        strides=[8, 8, 8],      # Tổng stride = 4*4*8*4 = 512 (Hoặc chỉnh tùy ý bạn muốn)
        embed_dim=768
    ):
        super().__init__()
        
        # 1. Stem: Initial features
        # Dùng kernel nhỏ hơn chút để giữ chi tiết ban đầu
        self.downsample_layers = nn.ModuleList() 
        stem = nn.Sequential(
            nn.Conv1d(in_channel, dims[0], kernel_size=7, padding=3),
            nn.LayerNorm(dims[0], eps=1e-6) # Chuẩn hóa ngay đầu vào (Channel-last handled by transpose logic if needed, but here usually (B, C, T))
        )
        # Lưu ý: LayerNorm mặc định pytorch expect (..., C). Conv1d ra (B, C, T).
        # Để đơn giản và khớp ConvNeXtBlock, Stem ta chỉ dùng Conv1d thôi, hoặc dùng GroupNorm.
        # Sửa lại Stem chuẩn ConvNeXt:
        self.stem = nn.Conv1d(in_channel, dims[0], kernel_size=7, padding=3)

        self.stages = nn.ModuleList()
        
        # Tạo các stage: Downsample -> N x ConvNeXtBlock
        curr_dim = dims[0]
        
        # Chúng ta sẽ có len(strides) stages.
        # Nếu dims và depths dài hơn, ta sẽ khớp theo len(strides)
        
        for i in range(len(strides)):
            # 1. Downsampling layer
            # Nếu không phải stage đầu tiên, ta downsample từ dim trước đó
            input_dim = curr_dim if i == 0 else dims[i-1]
            target_dim = dims[i]
            
            # Layer downsample riêng biệt
            downsample = nn.Sequential(
                nn.GroupNorm(1, input_dim), # Ổn định trước khi downsample
                nn.Conv1d(input_dim, target_dim, kernel_size=strides[i]*2, stride=strides[i], padding=strides[i]//2),
            )
            
            # 2. Các lớp ConvNeXt Block (Processing)
            # Đây là nơi "Sức mạnh" được sinh ra
            blocks = nn.Sequential(*[
                ConvNeXtBlock(
                    dim=target_dim, 
                    intermediate_dim=target_dim * 4, # Ratio 4 chuẩn của ConvNeXt
                    layer_scale_init_value=1e-6,
                    dw_kernel_size=7
                )
                for _ in range(depths[i])
            ])
            
            self.stages.append(nn.ModuleList([downsample, blocks]))
            curr_dim = target_dim

        # Final Projection
        self.final_norm = nn.LayerNorm(curr_dim)
        self.final_proj = nn.Linear(curr_dim, embed_dim)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x: (B, 1, T)
        x = self.stem(x)
        
        for downsample, blocks in self.stages:
            x = downsample(x)
            x = blocks(x)
        
        # Output: (B, C, T_downsampled)
        # Chuyển sang channel-last để qua LayerNorm cuối và Linear
        x = x.transpose(1, 2) # (B, T, C)
        x = self.final_norm(x)
        x = self.final_proj(x)
        
        # Trả về format (B, Embed, T) cho Quantizer
        return x.transpose(1, 2)