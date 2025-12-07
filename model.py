"""
Keyword Transformer Model - Exact Architecture from Your Training Code
Uses einops for efficient tensor rearrangement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# === Transformer Submodules ===
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k) * self.scale
        attn = self.attend(dots)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]) for _ in range(depth)
        ])

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


# === Keyword Transformer Model ===
class KeywordTransformer(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, dropout=0.1, emb_dropout=0.1):
        """
        Keyword Spotting Transformer model.
        
        Args:
            image_size (tuple): (height, width) of the spectrogram input. (40, 101)
            patch_size (tuple): (height, width) of each patch. (40, 10)
            num_classes (int): Number of keyword classes. 30
            dim (int): Embedding dimension. 160
            depth (int): Number of transformer layers. 6
            heads (int): Number of attention heads. 8
            mlp_dim (int): Hidden dimension for the feed-forward network. 256
            dropout (float): Dropout probability.
            emb_dropout (float): Dropout probability for embeddings.
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size

        # Compute number of patches
        num_patches = (image_size[1] + (patch_size[1] - 1)) // patch_size[1]
        patch_dim = image_size[0] * patch_size[1]  # (num_mels * patch_size_width)

        self.to_patch_embedding = nn.Linear(patch_dim, dim)
        self.rearrange = Rearrange("b c f (t p2) -> b t (c f p2)", p2=patch_size[1])

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head=dim // heads, mlp_dim=mlp_dim, dropout=dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        """
        Forward pass for the Keyword Transformer.
        Args:
            x (Tensor): Input tensor of shape (batch, channels, height, width).
                       For audio: (batch, 1, 40, 101) - (batch, channels, mel_bins, time_frames)
        Returns:
            Tensor: Predicted logits of shape (batch, num_classes).
        """
        b, c, f, t = x.shape
        pad_len = (self.patch_size[1] - (t % self.patch_size[1])) % self.patch_size[1]
        x = F.pad(x, (0, pad_len))  # Pad along time axis

        x = self.rearrange(x)  # Convert into patches
        x = self.to_patch_embedding(x)

        b, n, _ = x.shape
        cls_tokens = repeat(self.cls_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        x = x[:, 0]  # CLS token output
        return self.mlp_head(x)


def load_model(model_path, device='cpu'):
    """Load trained model from checkpoint with correct architecture"""
    model = KeywordTransformer(
        image_size=(40, 101),
        patch_size=(40, 10),
        num_classes=30,
        dim=160,
        depth=6,
        heads=8,
        mlp_dim=256,
        dropout=0.4
    )
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    
    return model
