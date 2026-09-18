import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiScaleElaboration(nn.Module):
    def __init__(self, growth=24, name='ms_block', device='cuda'):
        super().__init__()
        self.name = name
        self.growth = growth
        self.built = False
        self.device = device

    def _build(self, in_channels):
        self.layers = nn.ModuleDict({
            f'{self.name}_conv1': nn.Sequential(
                nn.Conv3d(in_channels, 12, kernel_size=1, stride=1, padding=0, bias=False),
                nn.InstanceNorm3d(12, affine=True, eps=1e-3),
                nn.LeakyReLU(0.3)
            ),
            f'{self.name}_conv2': nn.Sequential(
                nn.Conv3d(in_channels, 12, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm3d(12, affine=True, eps=1e-3),
                nn.LeakyReLU(0.3)
            ),
            f'{self.name}_conv3': nn.Sequential(
                nn.Conv3d(in_channels, 12, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm3d(12, affine=True, eps=1e-3),
                nn.LeakyReLU(0.3)
            ),
            f'{self.name}_conv3_1': nn.Sequential(
                nn.Conv3d(12, 12, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm3d(12, affine=True, eps=1e-3),
                nn.LeakyReLU(0.3)
            ),

            f'{self.name}_conv4': nn.Sequential(
                nn.Conv3d(in_channels, 12, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm3d(12, affine=True, eps=1e-3),
                nn.LeakyReLU(0.3)
            ),
            f'{self.name}_conv4_1': nn.Sequential(
                nn.Conv3d(12, 12, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm3d(12, affine=True, eps=1e-3),
                nn.LeakyReLU(0.3)
            ),
            f'{self.name}_conv4_2': nn.Sequential(
                nn.Conv3d(12, 12, kernel_size=3, stride=1, padding=1, bias=False),
                nn.InstanceNorm3d(12, affine=True, eps=1e-3),
                nn.LeakyReLU(0.3)
            ),

            f'{self.name}_conv5': nn.Conv3d(48, self.growth, kernel_size=1, stride=1, padding=0, bias=False) # was 48, increased to 64 for CT
        }).to(self.device)
        self.built = True

    def forward(self, x):
        if not self.built:
            self._build(x.shape[1])
        out1 = self.layers[f'{self.name}_conv1'](x)
        out2 = self.layers[f'{self.name}_conv2'](x)
        out3 = self.layers[f'{self.name}_conv3'](x)
        out3 = self.layers[f'{self.name}_conv3_1'](out3)
        out4 = self.layers[f'{self.name}_conv4'](x)
        out4 = self.layers[f'{self.name}_conv4_1'](out4)
        out4 = self.layers[f'{self.name}_conv4_2'](out4)
        concat = torch.cat([out1, out2, out3, out4], dim=1)
        out5 = self.layers[f'{self.name}_conv5'](concat)
        return torch.cat([x, out5], dim=1)


class Reduction(nn.Module):
    def __init__(self, name='reduction', device='cuda'):
        super().__init__()
        self.name = name
        self.built = False
        self.device = device

    def _build(self, in_channels):
        self.layers = nn.ModuleDict({
            f'{self.name}_conv_branch': nn.Sequential(
                nn.Conv3d(in_channels, 8, kernel_size=2, stride=2, padding=0, bias=True),
                nn.InstanceNorm3d(8, affine=True, eps=1e-3),
                nn.LeakyReLU(0.3, inplace=True)
            ),
            f'{self.name}_conv_final': nn.Sequential(
                nn.Conv3d(in_channels*2 + 8, in_channels, kernel_size=1, stride=1, padding=0, bias=True),
                nn.InstanceNorm3d(in_channels, affine=True, eps=1e-3),
                nn.LeakyReLU(0.3, inplace=True)
            )
        }).to(self.device)
        self.built = True

    def forward(self, x):
        if not self.built:
            self._build(x.shape[1])
        max_pool = F.max_pool3d(x, kernel_size=2, stride=2)
        avg_pool = F.avg_pool3d(x, kernel_size=2, stride=2)
        conv = self.layers[f'{self.name}_conv_branch'](x)
        concat = torch.cat([max_pool, avg_pool, conv], dim=1)
        out = self.layers[f'{self.name}_conv_final'](concat)
        return out


class Expansion(nn.Module):
    def __init__(self, name='expansion', device='cuda'):
        super().__init__()
        self.name = name
        self.built = False
        self.device = device

    def _build(self, in_channels):
        channel = in_channels // 2
        self.layers = nn.ModuleDict({
            f'{self.name}_conv1': nn.Sequential(
                nn.Conv3d(in_channels, channel, kernel_size=1, stride=1, padding=0, bias=True),
                nn.InstanceNorm3d(channel, affine=True, eps=1e-3),
                nn.LeakyReLU(0.3, inplace=True)
            ),
            f'{self.name}_conv_transpose': nn.ConvTranspose3d(
                channel, channel, kernel_size=3, stride=2, padding=1, output_padding=1, bias=True
            ),
            f'{self.name}_conv2': nn.Sequential(
                nn.Conv3d(channel*2, channel, kernel_size=1, stride=1, padding=0, bias=True),
                nn.InstanceNorm3d(channel, affine=True, eps=1e-3),
                nn.LeakyReLU(0.3, inplace=True)
            )
        }).to(self.device)   
        self.built = True

    def forward(self, x):
        if not self.built:
            self._build(x.shape[1])
        conv1 = self.layers[f'{self.name}_conv1'](x)
        up_sample = F.interpolate(conv1, scale_factor=2, mode='trilinear', align_corners=False)
        conv_transpose = self.layers[f'{self.name}_conv_transpose'](conv1)
        concat = torch.cat([up_sample, conv_transpose], dim=1)
        out = self.layers[f'{self.name}_conv2'](concat)
        return out



class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1, alpha=0.3):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.device = device   
        
        # LayerNorms
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        
        # Multi-Head Attention
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.LeakyReLU(alpha, inplace=True),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.LeakyReLU(alpha, inplace=True),
            nn.Dropout(dropout)
        )
        
        # Dropout for attention output
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """
        x: [B, N, dim]   (batch, sequence/patches, embedding dim)
        attn_mask: optional mask of shape [B, N, N] or [N, N]
        """
        # LayerNorm + MHA
        x_norm = self.norm1(x)
        # MultiheadAttention expects (B, N, dim) if batch_first=True
        attn_output, attn_scores = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask, need_weights=True)
        attn_output = self.attn_dropout(attn_output)
        
        # Residual
        x = x + attn_output
        
        # LayerNorm + MLP
        x_norm2 = self.norm2(x)
        mlp_output = self.mlp(x_norm2)
        
        # Residual
        x = x + mlp_output
        
        return x, attn_scores

