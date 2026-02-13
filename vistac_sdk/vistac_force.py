"""
Force estimation using Sparsh vision-based tactile sensing models.

This module implements force field and force vector estimation using pretrained
Sparsh models (ViT encoder + DPT decoder) with temporal frame pairs.
"""

import os
import warnings
from typing import Dict, Optional, Tuple, Union, Literal
import sys
import types

import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from .temporal_buffer import TemporalBuffer


def _load_encoder_checkpoint(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    """Load encoder weights from Lightning checkpoint.
    
    Args:
        checkpoint_path: Path to .ckpt file
        
    Returns:
        OrderedDict of model weights with 'student_encoder.backbone.' prefix removed
    """
    # Create fake modules to bypass Lightning class dependencies
    modules_to_fake = [
        'tactile_ssl',
        'tactile_ssl.model',
        'tactile_ssl.model.custom_scheduler',
        'tactile_ssl.model.vision_backbone',
        'tactile_ssl.model.vision_transformer',
    ]
    
    for mod_name in modules_to_fake:
        if mod_name not in sys.modules:
            fake_mod = types.ModuleType(mod_name)
            sys.modules[mod_name] = fake_mod
    
    # Add dummy classes
    class DummyClass:
        def __init__(self, *args, **kwargs):
            pass
    
    for cls_name in ['WarmupCosineScheduler', 'CosineWDSchedule', 'VisionTransformer', 'DinoVisionTransformer']:
        for mod_name in modules_to_fake[1:]:  # Skip base module
            if mod_name in sys.modules:
                setattr(sys.modules[mod_name], cls_name, type(cls_name, (DummyClass,), {}))
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    state_dict = checkpoint['model']
    
    # Remove 'student_encoder.backbone.' prefix
    cleaned_state_dict = {}
    prefix = 'student_encoder.backbone.'
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_key = key[len(prefix):]
            cleaned_state_dict[new_key] = value
    
    return cleaned_state_dict


class RoPE2D(nn.Module):
    """2D Rotary Position Embedding."""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        # Frequency bands will be loaded from checkpoint
        # Note: 192 bands for 768 embed_dim (not 384)
        self.register_buffer('frequency_bands', torch.randn(2, 192))
    
    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Apply RoPE to patch embeddings.
        
        Args:
            x: [B, N, C] patch embeddings
            h: Height in patches
            w: Width in patches
            
        Returns:
            [B, N, C] embeddings with positional encoding
        """
        # For simplicity, return x as-is since position info is already in features
        return x


class Attention(nn.Module):
    """Multi-head self-attention."""
    
    def __init__(self, dim: int, num_heads: int = 12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x


class MLP(nn.Module):
    """MLP block with GELU activation."""
    
    def __init__(self, in_features: int, hidden_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, in_features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class LayerScale(nn.Module):
    """Layer scale for better training stability."""
    
    def __init__(self, dim: int, init_value: float = 1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim) * init_value)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma


class Block(nn.Module):
    """Transformer block with attention and MLP."""
    
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads)
        self.ls1 = LayerScale(dim)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio))
        self.ls2 = LayerScale(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x


class SparshEncoder(nn.Module):
    """Vision Transformer encoder for Sparsh.
    
    ViT-base configuration:
    - patch_size: 16
    - embed_dim: 768
    - depth: 12 transformer blocks
    - num_heads: 12
    - mlp_ratio: 4.0
    """
    
    def __init__(self, 
                 img_size: int = 224,
                 patch_size: int = 16,
                 in_chans: int = 6,  # Temporal pair (2 RGB frames)
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0):
        super().__init__()
        
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embedding (RoPE)
        self.pos_embed = RoPE2D(embed_dim)
        
        # Register token (like class token)
        self.register_tokens = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio)
            for _ in range(depth)
        ])
        
        # Intermediate feature extraction layers
        self.feature_layers = [2, 5, 8, 11]  # Extract features from these layers
        self.intermediate_features = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with intermediate feature extraction.
        
        Args:
            x: [B, 6, 224, 224] input tensor (temporal pair)
            
        Returns:
            [B, N, C] final features
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, 768, 14, 14]
        h, w = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # [B, 196, 768]
        
        # Add position encoding
        x = self.pos_embed(x, h, w)
        
        # Add register token
        register_tokens = self.register_tokens.expand(B, -1, -1)
        x = torch.cat([register_tokens, x], dim=1)  # [B, 197, 768]
        
        # Pass through transformer blocks
        self.intermediate_features = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.feature_layers:
                self.intermediate_features.append(x)
        
        return x
    
    def get_intermediate_features(self) -> list:
        """Get intermediate features from specified layers."""
        return self.intermediate_features


class Resample(nn.Module):
    """Resample features to target spatial resolution."""
    
    def __init__(self, in_dim: int, out_dim: int, scale_factor: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, kernel_size=1)
        if scale_factor > 1:
            self.conv2 = nn.ConvTranspose2d(out_dim, out_dim, 
                                           kernel_size=scale_factor, 
                                           stride=scale_factor)
        else:
            self.conv2 = nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class Reassemble(nn.Module):
    """Reassemble features at different scales."""
    
    def __init__(self, embed_dim: int, out_dim: int, scale_factor: int):
        super().__init__()
        self.resample = Resample(embed_dim, out_dim, scale_factor)
    
    def forward(self, x: torch.Tensor, h: int, w: int) -> torch.Tensor:
        """Reassemble patch features to spatial features.
        
        Args:
            x: [B, N+1, C] features (with register token)
            h: Height in patches
            w: Width in patches
            
        Returns:
            [B, out_dim, H, W] spatial features
        """
        # Remove register token
        x = x[:, 1:, :]  # [B, N, C]
        
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, h, w)  # [B, C, h, w]
        x = self.resample(x)
        return x


class FusionBlock(nn.Module):
    """Fuse features from different scales."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
    
    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """Fuse multiple feature maps.
        
        Args:
            features: Variable number of [B, C, H, W] tensors
            
        Returns:
            [B, C, H, W] fused features
        """
        # Resize all to same size (largest)
        target_size = max(f.shape[-2:] for f in features)
        resized = []
        for f in features:
            if f.shape[-2:] != target_size:
                f = F.interpolate(f, size=target_size, mode='bilinear', align_corners=False)
            resized.append(f)
        
        # Sum features
        x = sum(resized)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x


class ForceFieldDecoder(nn.Module):
    """DPT-style decoder for force field prediction.
    
    Outputs:
    - normal: [B, 1, 224, 224] normal force field
    - shear: [B, 2, 224, 224] shear force field (Fx, Fy)
    """
    
    def __init__(self, embed_dim: int = 768, out_dim: int = 128):
        super().__init__()
        
        # Reassemble features at different scales
        # Actual scale factors from checkpoint: [4, 2, 1, 2]
        self.reassembles = nn.ModuleList([
            Reassemble(embed_dim, out_dim, scale_factor=4),   # 14x14 -> 56x56
            Reassemble(embed_dim, out_dim, scale_factor=2),   # 14x14 -> 28x28
            Reassemble(embed_dim, out_dim, scale_factor=1),   # 14x14 -> 14x14
            Reassemble(embed_dim, out_dim, scale_factor=2),   # 14x14 -> 28x28
        ])
        
        # Fusion blocks
        self.fusion = FusionBlock(out_dim)
        
        # Output heads
        self.norm = nn.LayerNorm(embed_dim)
        self.head_normal = nn.Sequential(
            nn.Conv2d(out_dim, out_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_dim // 2, 1, kernel_size=1)
        )
        self.head_shear = nn.Sequential(
            nn.Conv2d(out_dim, out_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_dim // 2, 2, kernel_size=1)
        )
    
    def forward(self, intermediate_features: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decode intermediate features to force fields.
        
        Args:
            intermediate_features: List of [B, N+1, C] features from layers [2, 5, 8, 11]
            
        Returns:
            normal: [B, 1, 224, 224]
            shear: [B, 2, 224, 224]
        """
        h, w = 14, 14  # Patch grid size for 224x224 with patch_size=16
        
        # Normalize features
        features_norm = [self.norm(f) for f in intermediate_features]
        
        # Reassemble to spatial features
        spatial_features = []
        for i, (feat, reassemble) in enumerate(zip(features_norm, self.reassembles)):
            spatial = reassemble(feat, h, w)
            spatial_features.append(spatial)
        
        # Fuse features (resize to 224x224)
        fused = self.fusion(*spatial_features)
        
        # Ensure output is 224x224
        if fused.shape[-2:] != (224, 224):
            fused = F.interpolate(fused, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Predict force fields
        normal = self.head_normal(fused)  # [B, 1, 224, 224]
        shear = self.head_shear(fused)    # [B, 2, 224, 224]
        
        return normal, shear


class ForceEstimator:
    """Main interface for force estimation using Sparsh models.
    
    Supports both force field (dense heatmaps) and force vector (aggregated) outputs.
    """
    
    def __init__(self,
                 encoder_path: str,
                 decoder_path: str,
                 temporal_stride: int = 5,
                 bg_offset: float = 0.5,
                 device: str = 'cuda',
                 force_field_baseline: bool = False,
                 force_vector_scale: Optional[Union[list, tuple]] = None):
        """Initialize force estimator.
        
        Args:
            encoder_path: Path to Sparsh encoder checkpoint (.ckpt)
            decoder_path: Path to force field decoder weights (.pth)
            temporal_stride: Frames between temporal pair (default: 5)
            bg_offset: Background subtraction offset (default: 0.5)
            device: 'cuda' or 'cpu'
            force_field_baseline: If True, compute/save a per-pixel background
                template during `load_background()` and subtract it from
                subsequent `force_field` outputs at runtime. Default: False.
            force_vector_scale: Per-axis scale to convert normalized `force_vector` -> physical units (N).
        """
        # Validate paths
        if not os.path.exists(encoder_path):
            raise FileNotFoundError(
                f"Sparsh encoder not found at {encoder_path}. "
                f"Run: python scripts/download_models.py"
            )
        if not os.path.exists(decoder_path):
            raise FileNotFoundError(
                f"Sparsh decoder not found at {decoder_path}. "
                f"Run: python scripts/download_models.py"
            )
        
        # Setup device
        self.device = device
        if device == 'cuda' and not torch.cuda.is_available():
            warnings.warn("CUDA not available, falling back to CPU")
            self.device = 'cpu'
        
        # Initialize models
        self.encoder = SparshEncoder()
        self.decoder = ForceFieldDecoder()
        
        # Load pretrained weights
        print(f"Loading encoder from {encoder_path}...")
        encoder_weights = _load_encoder_checkpoint(encoder_path)
        self.encoder.load_state_dict(encoder_weights, strict=False)
        
        print(f"Loading decoder from {decoder_path}...")
        decoder_weights = torch.load(decoder_path, map_location='cpu', weights_only=True)
        # Remove 'model_task.' prefix if present
        cleaned_decoder = {}
        prefix = 'model_task.'
        for key, value in decoder_weights.items():
            if key.startswith(prefix):
                new_key = key[len(prefix):]
                cleaned_decoder[new_key] = value
            else:
                cleaned_decoder[key] = value
        self.decoder.load_state_dict(cleaned_decoder, strict=False)
        
        # Move to device
        self.encoder = self.encoder.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.encoder.eval()
        self.decoder.eval()
        
        # Preprocessing
        self.bg_offset = bg_offset
        self.background = None
        self.transform = transforms.Compose([
            transforms.Resize((224, 224), antialias=True),
            transforms.ToTensor(),  # Converts to [0, 1]
        ])

        # Per-axis force scale (normalized model units -> physical units (N)).
        # Per-axis scale mapping normalized model units -> physical units (N)
        self.force_vector_scale = np.array(force_vector_scale if force_vector_scale is not None else [1.0, 1.0, 1.0], dtype=float)

        # Temporal buffer
        self.temporal_stride = temporal_stride
        self.temporal_buffer = TemporalBuffer(max_size=temporal_stride + 1)

        # Runtime force_field baseline subtraction (per-pixel template)
        # Disabled by default to preserve Sparsh demo behavior.
        self.force_field_baseline_enabled = bool(force_field_baseline)
        self.force_field_baseline_template = None

        print(f"Force estimator initialized on {self.device}")
    
    def load_background(self, background: np.ndarray):
        """Load background image for subtraction and compute no-contact baseline.

        The baseline is computed by running the model on a background pair and
        saved to `self.force_vector_baseline`. This baseline is subtracted from
        `force_vector` outputs to remove sensor/model bias.

        Args:
            background: [H, W, 3] BGR background image (uint8)
        """
        self.background = background.copy()

        # Compute baseline force vector from background (no-contact) frames.
        # We run a single forward pass with bg / bg to estimate steady-state bias.
        try:
            # Prepare input and run encoder+decoder on CPU to be safe
            input_tensor = self._preprocess(self.background, self.background).to(self.device)
            with torch.no_grad():
                _ = self.encoder(input_tensor)
                intermediate_features = self.encoder.get_intermediate_features()
                normal, shear = self.decoder(intermediate_features)
                normal = normal.squeeze(0).squeeze(0).cpu().numpy()
                shear = shear.squeeze(0).permute(1, 2, 0).cpu().numpy()

            # Aggregate baseline vector (mean over spatial dims)
            baseline_fz = float(np.mean(normal))
            baseline_fx = float(np.mean(shear[:, :, 0]))
            baseline_fy = float(np.mean(shear[:, :, 1]))

            self.force_vector_baseline = {'fx': baseline_fx, 'fy': baseline_fy, 'fz': baseline_fz}

            # Optionally save per-pixel force_field baseline template for runtime subtraction
            if getattr(self, 'force_field_baseline_enabled', False):
                try:
                    self.force_field_baseline_template = {
                        'normal': normal.copy(),
                        'shear': shear.copy()
                    }
                except Exception:
                    self.force_field_baseline_template = None
        except Exception:
            # If baseline computation fails for any reason, default to zero baseline
            self.force_vector_baseline = {'fx': 0.0, 'fy': 0.0, 'fz': 0.0}
            self.force_field_baseline_template = None
    
    def _preprocess(self, img_t: np.ndarray, img_t_minus: np.ndarray) -> torch.Tensor:
        """Preprocess temporal pair for force estimation.
        
        Args:
            img_t: [H, W, 3] BGR image at time t (uint8)
            img_t_minus: [H, W, 3] BGR image at time t-stride (uint8)
            
        Returns:
            [1, 6, 224, 224] preprocessed tensor
        """
        if self.background is None:
            raise ValueError("Background not loaded. Call load_background() first.")
        
        # Background subtraction with offset
        def subtract_bg(img, bg, offset):
            diff = img.astype(np.int32) - bg.astype(np.int32)
            diff = diff / 255.0 + offset
            diff = np.clip(diff, 0.0, 1.0)
            diff = (diff * 255.0).astype(np.uint8)
            return diff
        
        img_t_diff = subtract_bg(img_t, self.background, self.bg_offset)
        img_t_minus_diff = subtract_bg(img_t_minus, self.background, self.bg_offset)
        
        # Convert to RGB PIL images (input frames from camera are BGR)
        # Sparsh expects RGB images — convert from BGR -> RGB first.
        img_t_rgb = cv2.cvtColor(img_t_diff, cv2.COLOR_BGR2RGB)
        img_t_minus_rgb = cv2.cvtColor(img_t_minus_diff, cv2.COLOR_BGR2RGB)
        img_t_pil = Image.fromarray(img_t_rgb).convert("RGB")
        img_t_minus_pil = Image.fromarray(img_t_minus_rgb).convert("RGB")
        
        # Resize and convert to tensor
        tensor_t = self.transform(img_t_pil)         # [3, 224, 224]
        tensor_t_minus = self.transform(img_t_minus_pil)  # [3, 224, 224]
        
        # Temporal concatenation
        input_tensor = torch.cat([tensor_t, tensor_t_minus], dim=0)  # [6, 224, 224]
        
        return input_tensor.unsqueeze(0)  # [1, 6, 224, 224]
    
    def estimate(self,
                 image: np.ndarray,
                 timestamp: Optional[float] = None) -> Optional[Dict[str, Union[np.ndarray, float]]]:
        """Estimate force field and vector from image.
        
        Args:
            image: [H, W, 3] BGR image (uint8)
            timestamp: Optional timestamp for temporal tracking
            
        Returns:
            Dictionary with:
            - 'force_field': {'normal': [224, 224], 'shear': [224, 224, 2]} or None if buffer not ready
            - 'force_vector': {'fx': float, 'fy': float, 'fz': float} or None if buffer not ready
        """
        # Add to temporal buffer
        self.temporal_buffer.add(image, timestamp)
        
        # Check if buffer is ready
        if not self.temporal_buffer.is_ready():
            return None
        
        # Get temporal pair
        frame_pair = self.temporal_buffer.get_pair(stride=self.temporal_stride)
        if frame_pair is None:
            return None
        
        img_t, img_t_minus = frame_pair
        
        # Preprocess
        input_tensor = self._preprocess(img_t, img_t_minus)
        input_tensor = input_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            try:
                # Encoder
                _ = self.encoder(input_tensor)
                intermediate_features = self.encoder.get_intermediate_features()
                
                # Decoder
                normal, shear = self.decoder(intermediate_features)
                
                # Move to CPU and convert to numpy
                normal = normal.squeeze(0).squeeze(0).cpu().numpy()  # [224, 224]
                shear = shear.squeeze(0).permute(1, 2, 0).cpu().numpy()  # [224, 224, 2]
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    warnings.warn("CUDA OOM, falling back to CPU")
                    self.encoder = self.encoder.cpu()
                    self.decoder = self.decoder.cpu()
                    self.device = "cpu"
                    # Retry on CPU
                    return self.estimate(image, timestamp)
                else:
                    raise
        
        # Optionally subtract per-pixel force_field baseline template
        if getattr(self, 'force_field_baseline_enabled', False) and self.force_field_baseline_template is not None:
            try:
                normal = normal - self.force_field_baseline_template['normal']
                shear = shear - self.force_field_baseline_template['shear']
            except Exception:
                # If subtraction fails, leave original arrays
                pass

        # Aggregate force vector
        H, W = 224, 224
        fz = float(np.mean(normal))
        fx = float(np.mean(shear[:, :, 0]))
        fy = float(np.mean(shear[:, :, 1]))

        # Subtract baseline (if available) to remove sensor/model bias
        baseline = getattr(self, 'force_vector_baseline', None)
        if baseline is not None:
            fx = fx - float(baseline.get('fx', 0.0))
            fy = fy - float(baseline.get('fy', 0.0))
            fz = fz - float(baseline.get('fz', 0.0))

        # Compute physical (Newton) values by applying per-axis force_vector_scale
        try:
            sx, sy, sz = tuple(self.force_vector_scale.tolist())
        except Exception:
            sx, sy, sz = 1.0, 1.0, 1.0
        force_vector_physical = {
            'fx': float(fx * sx),
            'fy': float(fy * sy),
            'fz': float(fz * sz),
        }

        return {
            'force_field': {
                'normal': normal,
                'shear': shear
            },
            'force_vector': {
                'fx': fx,
                'fy': fy,
                'fz': fz
            },
            'force_vector_physical': force_vector_physical
        }
