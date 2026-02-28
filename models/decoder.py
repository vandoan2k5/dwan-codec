import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional

# SPECTRAL_OPS  ---------------------------------------------
class ISTFT(nn.Module):
    """
    Custom implementation of ISTFT since torch.istft doesn't allow custom padding (other than `center=True`) with
    windowing. This is because the NOLA (Nonzero Overlap Add) check fails at the edges.
    See issue: https://github.com/pytorch/pytorch/issues/62323
    Specifically, in the context of neural vocoding we are interested in "same" padding analogous to CNNs.
    The NOLA constraint is met as we trim padded samples anyway.

    Args:
        n_fft (int): Size of Fourier transform.
        hop_length (int): The distance between neighboring sliding window frames.
        win_length (int): The size of window frame and STFT filter.
        padding (str, optional): Type of padding. Options are "center" or "same". Defaults to "same".
    """

    def __init__(self, n_fft: int, hop_length: int, win_length: int, padding: str = "same"):
        super().__init__()
        if padding not in ["center", "same"]:
            raise ValueError("Padding must be 'center' or 'same'.")
        self.padding = padding
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        window = torch.hann_window(win_length)
        self.register_buffer("window", window)

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Compute the Inverse Short Time Fourier Transform (ISTFT) of a complex spectrogram.

        Args:
            spec (Tensor): Input complex spectrogram of shape (B, N, T), where B is the batch size,
                            N is the number of frequency bins, and T is the number of time frames.

        Returns:
            Tensor: Reconstructed time-domain signal of shape (B, L), where L is the length of the output signal.
        """
        if self.padding == "center":
            spec[:,0] = 0 # fixes some strange bug where first/last freqs don't matter when bs<16 which causes exploding gradients
            spec[:,-1] = 0
            # Fallback to pytorch native implementation
            return torch.istft(spec, self.n_fft, self.hop_length, self.win_length, self.window, center=True)
        elif self.padding == "same":
            pad = (self.win_length - self.hop_length) // 2
        else:
            raise ValueError("Padding must be 'center' or 'same'.")

        assert spec.dim() == 3, "Expected a 3D tensor as input"
        B, N, T = spec.shape

        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]

        # Overlap and Add
        output_size = (T - 1) * self.hop_length + self.win_length
        y = torch.nn.functional.fold(
            ifft, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]

        # Window envelope
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = torch.nn.functional.fold(
            window_sq, output_size=(1, output_size), kernel_size=(1, self.win_length), stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]

        # Normalize
        assert (window_envelope > 1e-11).all()
        y = y / window_envelope

        return y


# MODULES       ---------------------------------------------
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


# MODELS        ---------------------------------------------
class VocosBackbone(nn.Module):
    """
    Vocos backbone module built with ConvNeXt blocks. Supports additional conditioning with Adaptive Layer Normalization

    Args:
        input_channels (int): Number of input features channels.
        dim (int): Hidden dimension of the model.
        intermediate_dim (int): Intermediate dimension used in ConvNeXtBlock.
        num_layers (int): Number of ConvNeXtBlock layers.
        layer_scale_init_value (float, optional): Initial value for layer scaling. Defaults to `1 / num_layers`.
    """

    def __init__(
        self,
        input_channels: int,
        dim: int,
        intermediate_dim: int,
        num_layers: int,
        input_kernel_size: int = 9,
        dw_kernel_size: int = 9,
        layer_scale_init_value: Optional[float] = None,
        pad: str = 'zeros',
    ):
        super().__init__()
        self.embed = nn.Conv1d(input_channels, dim, kernel_size=input_kernel_size, padding=input_kernel_size//2, padding_mode=pad)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.convnext = nn.ModuleList(
            [
                ConvNeXtBlock(
                    dim=dim,
                    intermediate_dim=intermediate_dim,
                    dw_kernel_size=dw_kernel_size,
                    layer_scale_init_value=layer_scale_init_value or 1 / num_layers**0.5,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_layer_norm = nn.LayerNorm(dim, eps=1e-6)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embed(x) # (B, C, L)
        x = self.norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        for conv_block in self.convnext:
            x = conv_block(x)
        x = self.final_layer_norm(x.transpose(1, 2))
        x = x.transpose(1, 2)
        return x


class ISTFTHead(nn.Module):
    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "center"):
        super().__init__()
        out_dim = n_fft + 2 # Magnitude + Phase
        self.out = nn.Linear(dim, out_dim)
        self.istft = ISTFT(n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding)
    @torch.compiler.disable
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x input: (B, C, L)
        x = self.out(x.transpose(1, 2)).transpose(1, 2)
        mag, p = x.chunk(2, dim=1)
        
        mag = torch.exp(mag)
        mag = torch.clip(mag, max=1e2) 
        
        # Tong hop tin hieu
        S = mag * (torch.cos(p) + 1j * torch.sin(p))
        audio = self.istft(S)
        return audio.unsqueeze(1)

class Decoder(nn.Module):
    def __init__(self, 
                 embed_dim=768, 
                 decoder_dim=768, 
                 decoder_num_layers=8,
                 upscale=4, 
                 n_fft=1024, 
                 hop_length=512,
                 dw_kernel=3):
        super().__init__()
        self.upscale = upscale
        
        self.decoder = VocosBackbone(
            input_channels=embed_dim,
            dim=decoder_dim,
            intermediate_dim=decoder_dim * 3,
            num_layers=decoder_num_layers,
            input_kernel_size=1,
            dw_kernel_size=dw_kernel,
        )
        
        self.head = ISTFTHead(
            dim=decoder_dim,
            n_fft=n_fft,
            hop_length=hop_length
        )

    def forward(self, x):
        T = x.size(2)
        x = self.decoder(x)  
        return self.head(x)