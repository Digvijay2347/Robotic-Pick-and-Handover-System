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
        use_goal_conditioning: bool = True,
        use_contact_conditioning: bool = True,
        goal_dim: int = 3,
        contact_dim: int = 2,
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
        self.use_goal_conditioning = use_goal_conditioning
        self.use_contact_conditioning = use_contact_conditioning
        self.goal_dim = goal_dim
        self.contact_dim = contact_dim

        # 1. Visual Backbones (one per camera)
        self.backbones = nn.ModuleDict({
            cam: VisualBackbone(d_model=d_model, spatial_grid=(2, 2), freeze=freeze_backbone)
            for cam in self.camera_names
        })
        self.num_cam_tokens = len(self.camera_names) * 4  # 3 cams * 4 tokens = 12

        # 2. Proprioception, Goal, Contact & Latent Projectors
        self.qpos_proj = nn.Linear(state_dim, d_model)
        self.latent_proj = nn.Linear(latent_dim, d_model)

        if use_goal_conditioning:
            self.goal_proj = nn.Sequential(
                nn.Linear(goal_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.goal_proj = None

        if use_contact_conditioning:
            self.contact_proj = nn.Sequential(
                nn.Linear(contact_dim, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
        else:
            self.contact_proj = None

        # 3. C-VAE Style Encoder (active during training)
        self.cvae = CVAEEncoder(
            action_dim=action_dim,
            state_dim=state_dim,
            d_model=d_model,
            latent_dim=latent_dim,
            nhead=nhead,
        )

        # 4. Transformer Encoder (Fused multimodal context)
        # Sequence length: 1 (latent) + 1 (qpos) + (1 if goal else 0) + (1 if contact else 0) + num_cam_tokens
        num_cond_tokens = (1 if use_goal_conditioning else 0) + (1 if use_contact_conditioning else 0)
        self.total_enc_tokens = 2 + num_cond_tokens + self.num_cam_tokens
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

    def load_pretrained_pure_act(self, checkpoint_path: str):
        """
        Warm-starts policy weights from a pure ACT model checkpoint, transferring
        visual backbones, C-VAE, transformer encoder/decoder layers, and action head.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

        filtered_dict = {}
        for k, v in state_dict.items():
            if k == "enc_pos_emb":
                continue
            if k in self.state_dict() and self.state_dict()[k].shape == v.shape:
                filtered_dict[k] = v

        msg = self.load_state_dict(filtered_dict, strict=False)
        print(f"[ACTPolicy] Warm-started {len(filtered_dict)} layers from {checkpoint_path}!")
        print(f"  Missing (new) keys: {msg.missing_keys}")

        if "enc_pos_emb" in state_dict and state_dict["enc_pos_emb"].shape[1] == 14 and self.total_enc_tokens == 16:
            with torch.no_grad():
                old_emb = state_dict["enc_pos_emb"]
                self.enc_pos_emb.data[:, 0:2, :] = old_emb[:, 0:2, :]
                self.enc_pos_emb.data[:, 4:16, :] = old_emb[:, 2:14, :]
            print("  [ACTPolicy] Successfully transferred positional embeddings into 16-token layout!")

    def forward(
        self,
        images: Dict[str, torch.Tensor],
        qpos: torch.Tensor,
        target_pos: Optional[torch.Tensor] = None,
        contact_states: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for both training and inference.

        Args:
            images: Dict of camera tensors (B, 3, H, W)
            qpos: Joint proprioception (B, 14)
            target_pos: Optional 3D centroid coordinates (B, 3)
            contact_states: Optional binary contact states (B, 2)
            actions: Optional target action chunk (B, K, 14) for C-VAE training
            is_pad: Optional padding mask (B, K)

        Returns:
            Dict containing predicted actions, latent stats, and computed losses.
        """
        b = qpos.shape[0]

        # 1. Obtain Latent style z
        if actions is not None:
            z, mu, logvar = self.cvae(actions, qpos)
        else:
            z = torch.zeros(b, self.latent_dim, device=qpos.device, dtype=qpos.dtype)
            mu, logvar = None, None

        # 2. Extract visual feature tokens across cameras
        cam_token_list = []
        for cam_name in self.camera_names:
            img = images[cam_name]
            cam_toks = self.backbones[cam_name](img)  # (B, 4, d_model)
            cam_token_list.append(cam_toks)
        visual_tokens = torch.cat(cam_token_list, dim=1)  # (B, 12, d_model)

        # 3. Project state, goal, contact & latent
        qpos_token = self.qpos_proj(qpos).unsqueeze(1)    # (B, 1, d_model)
        latent_token = self.latent_proj(z).unsqueeze(1)   # (B, 1, d_model)

        tokens_list = [latent_token, qpos_token]
        if self.use_goal_conditioning and self.goal_proj is not None:
            if target_pos is None:
                target_pos = torch.zeros(b, self.goal_dim, device=qpos.device, dtype=qpos.dtype)
            goal_token = self.goal_proj(target_pos).unsqueeze(1)
            tokens_list.append(goal_token)

        if self.use_contact_conditioning and self.contact_proj is not None:
            if contact_states is None:
                contact_states = torch.zeros(b, self.contact_dim, device=qpos.device, dtype=qpos.dtype)
            contact_token = self.contact_proj(contact_states).unsqueeze(1)
            tokens_list.append(contact_token)

        tokens_list.append(visual_tokens)

        # 4. Formulate Encoder Input Sequence
        enc_seq = torch.cat(tokens_list, dim=1)
        enc_seq = enc_seq + self.enc_pos_emb[:, :enc_seq.shape[1], :]

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
    def predict_action_chunk(
        self,
        images: Dict[str, torch.Tensor],
        qpos: torch.Tensor,
        target_pos: Optional[torch.Tensor] = None,
        contact_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Inference helper returning raw predicted action chunk (B, K, 14)."""
        self.eval()
        res = self.forward(
            images=images,
            qpos=qpos,
            target_pos=target_pos,
            contact_states=contact_states,
        )
        return res["pred_actions"]

