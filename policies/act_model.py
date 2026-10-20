"""
Action Chunking with Transformers (ACT) Model Architecture.
Implements ResNet-18 multi-view visual backbones, 14-DOF proprioception embedding,
C-VAE style latent encoder, and Transformer Encoder-Decoder action chunk predictor.
Based on Tony Zhao et al. (Stanford / Google, 2023).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from typing import Dict, List, Tuple, Optional


class VisualBackbone(nn.Module):
    """
    ResNet-18 feature extractor for a single camera view.
    Extracts spatial feature patches and projects to transformer d_model.
    """

    def __init__(self, d_model: int = 512, spatial_grid: Tuple[int, int] = (2, 2), freeze: bool = True):
        super().__init__()
        # Load standard ResNet-18 without classification head
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.freeze = freeze
        if freeze:
            for p in [self.conv1, self.bn1, self.layer1, self.layer2, self.layer3, self.layer4]:
                for param in p.parameters():
                    param.requires_grad = False

        self.adaptive_pool = nn.AdaptiveAvgPool2d(spatial_grid)
        self.proj = nn.Conv2d(512, d_model, kernel_size=1)
        self.num_patches = spatial_grid[0] * spatial_grid[1]
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Image tensor (B, 3, H, W)
        Returns:
            tokens: (B, num_patches, d_model)
        """
        if self.freeze:
            with torch.no_grad():
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

        x = self.adaptive_pool(x)  # (B, 512, gh, gw)
        x = self.proj(x)           # (B, d_model, gh, gw)
        b, d, gh, gw = x.shape
        x = x.flatten(2).permute(0, 2, 1)  # (B, gh*gw, d_model)
        return x


class CVAEEncoder(nn.Module):
    """
    Conditional Variational Autoencoder (C-VAE) Style Encoder.
    Encodes the target action chunk A conditioned on state qpos into latent z ~ N(mu, sigma^2).
    """

    def __init__(
        self,
        action_dim: int = 14,
        state_dim: int = 14,
        d_model: int = 512,
        latent_dim: int = 32,
        nhead: int = 8,
        num_layers: int = 2,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.d_model = d_model

        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.state_proj = nn.Linear(state_dim, d_model)
        self.action_proj = nn.Linear(action_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.latent_head = nn.Linear(d_model, 2 * latent_dim)

    def forward(self, actions: torch.Tensor, qpos: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            actions: Target action chunk (B, K, 14)
            qpos: Normalized joint proprioception (B, 14)
        Returns:
            z: Sampled latent style (B, latent_dim)
            mu: Latent distribution mean (B, latent_dim)
            logvar: Latent distribution log variance (B, latent_dim)
        """
        b = actions.shape[0]
        cls_tok = self.cls_token.expand(b, -1, -1)  # (B, 1, d_model)
        qpos_tok = self.state_proj(qpos).unsqueeze(1)  # (B, 1, d_model)
        act_toks = self.action_proj(actions)          # (B, K, d_model)

        seq = torch.cat([cls_tok, qpos_tok, act_toks], dim=1)  # (B, 2 + K, d_model)
        h = self.transformer(seq)
        cls_out = h[:, 0, :]  # (B, d_model)

        params = self.latent_head(cls_out)  # (B, 2 * latent_dim)
        mu = params[:, :self.latent_dim]
        logvar = params[:, self.latent_dim:]

        # Reparameterization trick
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar


class ACTPolicy(nn.Module):
    """
    Complete Action Chunking with Transformers (ACT) Policy.
    Maps multi-camera images and 14-DOF proprioception to action chunks of length K.
    """

    def __init__(
        self,
        camera_names: Optional[List[str]] = None,
        chunk_size: int = 25,
        state_dim: int = 14,
        action_dim: int = 14,
        d_model: int = 512,
        nhead: int = 8,
        enc_layers: int = 4,
        dec_layers: int = 4,
        dim_feedforward: int = 2048,
        latent_dim: int = 32,
        dropout: int = 0.1,
        kl_weight: float = 10.0,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.camera_names = camera_names or ["top", "wrist1", "wrist2"]
        self.chunk_size = chunk_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.freeze_backbone = freeze_backbone

        # 1. Visual Backbones (one per camera)
        self.backbones = nn.ModuleDict({
            cam: VisualBackbone(d_model=d_model, spatial_grid=(2, 2), freeze=freeze_backbone)
            for cam in self.camera_names
        })
        self.num_cam_tokens = len(self.camera_names) * 4  # 3 cams * 4 tokens = 12

        # 2. Proprioception & Latent Projectors
        self.qpos_proj = nn.Linear(state_dim, d_model)
        self.latent_proj = nn.Linear(latent_dim, d_model)

        # 3. C-VAE Style Encoder (active during training)
        self.cvae = CVAEEncoder(
            action_dim=action_dim,
            state_dim=state_dim,
            d_model=d_model,
            latent_dim=latent_dim,
            nhead=nhead,
        )

        # 4. Transformer Encoder (Fused multimodal context)
        # Sequence length: 1 (latent) + 1 (qpos) + num_cam_tokens
        self.total_enc_tokens = 2 + self.num_cam_tokens
        self.enc_pos_emb = nn.Parameter(torch.randn(1, self.total_enc_tokens, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(enc_layer, num_layers=enc_layers)

        # 5. Transformer Decoder (Action Chunk Synthesis)
        # Learnable positional queries for K action steps
        self.dec_query_emb = nn.Parameter(torch.randn(1, chunk_size, d_model) * 0.02)

        dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(dec_layer, num_layers=dec_layers)

        # 6. Action Prediction Head
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(
        self,
        images: Dict[str, torch.Tensor],
        qpos: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for both training and inference.

        Args:
            images: Dict of camera tensors (B, 3, H, W)
            qpos: Joint proprioception (B, 14)
            actions: Optional target action chunk (B, K, 14) for C-VAE training
            is_pad: Optional padding mask (B, K)

        Returns:
            Dict containing:
                "pred_actions": (B, K, 14)
                "mu": Latent mean (B, latent_dim) or None
                "logvar": Latent logvar (B, latent_dim) or None
                "total_loss": Scalar loss if actions provided
                "l1_loss": Scalar L1 action loss if actions provided
                "kl_loss": Scalar KL divergence loss if actions provided
        """
        b = qpos.shape[0]

        # 1. Obtain Latent style z
        if actions is not None:
            # Training mode: sample z from C-VAE encoder
            z, mu, logvar = self.cvae(actions, qpos)
        else:
            # Inference mode: set z to mean of prior (zeros)
            z = torch.zeros(b, self.latent_dim, device=qpos.device, dtype=qpos.dtype)
            mu, logvar = None, None

        # 2. Extract visual feature tokens across cameras
        cam_token_list = []
        for cam_name in self.camera_names:
            img = images[cam_name]
            cam_toks = self.backbones[cam_name](img)  # (B, 4, d_model)
            cam_token_list.append(cam_toks)
        visual_tokens = torch.cat(cam_token_list, dim=1)  # (B, 12, d_model)

        # 3. Project state and latent
        qpos_token = self.qpos_proj(qpos).unsqueeze(1)    # (B, 1, d_model)
        latent_token = self.latent_proj(z).unsqueeze(1)   # (B, 1, d_model)

        # 4. Formulate Encoder Input Sequence
        enc_seq = torch.cat([latent_token, qpos_token, visual_tokens], dim=1)  # (B, 14, d_model)
        enc_seq = enc_seq + self.enc_pos_emb

        # Pass through Transformer Encoder
        memory = self.transformer_encoder(enc_seq)  # (B, 14, d_model)

        # 5. Formulate Decoder Queries
        tgt = self.dec_query_emb.expand(b, -1, -1)  # (B, K, d_model)

        # Pass through Transformer Decoder
        decoded = self.transformer_decoder(tgt=tgt, memory=memory)  # (B, K, d_model)

        # 6. Action Head
        pred_actions = self.action_head(decoded)  # (B, K, 14)

        output = {
            "pred_actions": pred_actions,
            "mu": mu,
            "logvar": logvar,
        }

        # 7. Compute Loss if training
        if actions is not None:
            output.update(self.compute_loss(pred_actions, actions, is_pad, mu, logvar, kl_weight=self.kl_weight))

        return output

    def compute_loss(
        self,
        pred_actions: torch.Tensor,
        target_actions: torch.Tensor,
        is_pad: Optional[torch.Tensor] = None,
        mu: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
        kl_weight: float = 10.0,
    ) -> Dict[str, torch.Tensor]:
        """Computes masked L1 action chunk loss and KL divergence."""
        if is_pad is not None:
            mask = (~is_pad).unsqueeze(-1).float()  # (B, K, 1)
            l1_loss = (torch.abs(pred_actions - target_actions) * mask).sum() / mask.sum().clamp(min=1.0)
        else:
            l1_loss = F.l1_loss(pred_actions, target_actions)

        if mu is not None and logvar is not None:
            # KL(q(z|a,s) || N(0, I)) = -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
            kl_loss = -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        else:
            kl_loss = torch.tensor(0.0, device=pred_actions.device)

        total_loss = l1_loss + kl_weight * kl_loss

        return {
            "total_loss": total_loss,
            "l1_loss": l1_loss,
            "kl_loss": kl_loss,
        }

    @torch.no_grad()
    def predict_action_chunk(self, images: Dict[str, torch.Tensor], qpos: torch.Tensor) -> torch.Tensor:
        """Inference helper returning raw predicted action chunk (B, K, 14)."""
        self.eval()
        res = self.forward(images=images, qpos=qpos)
        return res["pred_actions"]
