"""
HOLIC.py
Modelul principal HOLIC — orchestrator cu injecție de dependențe.

Design:
  - HOLIC primește componentele deja construite prin constructor (AGREGARE),
    nu le creează el intern. Asta decuplează orchestratorul de implementări
    concrete și permite testare unitară + diversificarea relațiilor UML.
  - Constructorul alternativ `from_cfg(cfg, label_names, device)` este un
    factory care construiește componentele conform config-ului YAML și apoi
    apelează constructorul principal.

Pipeline training (Fig. 2 din HOLIC paper):
  1. Backbone (2G-GCN) extrage Z și logits
  2. MLP proiectează Z -> Z' (L2 normalizat, dim=512)
  3. Discriminatorul calculează scorurile MI între Z și G (Eq. 1)
  4. Similaritatea cosinus între Z' și T, cu temperature scaling (Eq. 3)
  5. Loss total: L = L_Label + λ1*L_Seg + λ2*L_MI + λ3*L_Cos (Eq. 4)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from omegaconf import DictConfig
from typing import Dict, Optional, List

from models.base import IBackbone, IProjection, ITextEncoder
from models.backbone import Backbone2GGCN
from models.clip_modules import (
    CLIPTextEncoder,
    MLPProjection,
    SimpleMLPProjection,
    IntegratedGlobalRepresentation,
    FeaturesCollector,
    Prototyping,
)
from models.discriminator import Discriminator


class HOLIC(nn.Module):
    """
    Video-based Human-Object Interaction recognition with CLIP Prior knowledge.

    Args (constructor principal — injecție de dependențe):
        backbone:        implementare IBackbone (ex. Backbone2GGCN)
        mlp_proj:        implementare IProjection (ex. MLPProjection)
        text_encoder:    implementare ITextEncoder (ex. CLIPTextEncoder)
        discriminator:   Discriminator MI
        global_rep:      IntegratedGlobalRepresentation (EMA G)
        num_classes:     număr de clase
        hidden_dim:      dimensiune ascunsă backbone
        clip_dim:        dimensiune feature CLIP
        label_names:     lista numelor claselor pentru template-uri
        clip_subject:    subiect în template CLIP ("person", "hand")
        clip_template:   template string(s) pentru encoder text
        device:          torch device
        use_learnable_prompts: activează C6b
        disable_geo_branch:    ablație — dezactivează ramura geometrică
        lambda4:               greutate L_PromptReg
    """

    def __init__(
        self,
        backbone: IBackbone,
        mlp_proj: IProjection,
        text_encoder: ITextEncoder,
        discriminator: Discriminator,
        global_rep: IntegratedGlobalRepresentation,
        num_classes: int,
        hidden_dim: int,
        clip_dim: int,
        label_names: List[str],
        clip_subject: str,
        clip_template,
        device: str = "cuda",
        use_learnable_prompts: bool = False,
        disable_geo_branch: bool = False,
        lambda4: float = 0.1,
    ):
        super().__init__()

        self.device = device
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.clip_dim = clip_dim
        self.use_learnable_prompts = use_learnable_prompts
        self.disable_geo_branch = disable_geo_branch

        # -----------------------------------------------------------------------
        # Componente injectate — relație UML: AGREGARE (diamant gol)
        # -----------------------------------------------------------------------
        self.backbone = backbone
        self.mlp_proj = mlp_proj
        self.discriminator = discriminator
        self.text_encoder = text_encoder
        self.global_rep = global_rep

        # Colector features Z' pentru EMA update la sfârșitul epocii
        # Compoziție acceptabilă — utilitar fără parametri proprii
        self.collector = FeaturesCollector()

        # Store label info
        self._label_names = label_names
        self._clip_subject = clip_subject
        self._clip_template = clip_template

        # -----------------------------------------------------------------------
        # Precompute frozen T — folosit pentru baseline și prompt reg loss
        # -----------------------------------------------------------------------
        frozen_T = self.text_encoder.precompute_frozen_T(
            label_names,
            subject=clip_subject,
            template=clip_template,
        )
        num_templates = len(clip_template) if isinstance(clip_template, (list, tuple)) else 1

        if self.use_learnable_prompts:
            self.register_buffer("T", frozen_T.clone())
        else:
            self.register_buffer("T", frozen_T)

        # C6b prompt regularization weight
        self._lambda4 = lambda4

        # Flag pentru modul de inferență
        self._inference_mode = False

    # ---------------------------------------------------------------------------
    # Factory: construiește HOLIC din config YAML (backward-compatible)
    # ---------------------------------------------------------------------------

    @classmethod
    def from_cfg(cls, cfg: DictConfig, label_names: List[str], device: str = "cuda"):
        """Constructor alternativ — creează componentele conform config-ului."""
        use_learnable_prompts = getattr(cfg.model, "use_learnable_prompts", False)
        disable_geo_branch = getattr(cfg.model, "disable_geo_branch", False)

        backbone = Backbone2GGCN(
            input_dim=cfg.data.roi_dim,
            hidden_dim=cfg.model.hidden_dim,
            num_classes=cfg.model.num_classes,
            num_layers=cfg.model.num_layers,
            dropout=cfg.model.dropout,
            geo_input_dim=getattr(cfg.data, "geo_input_dim", 4),
            C1=getattr(cfg.model, "geo_C1", 64),
            C2=getattr(cfg.model, "geo_C2", 128),
            boundary_kernel_size=getattr(cfg.model, "boundary_kernel_size", 5),
        )

        if getattr(cfg.model, "use_simple_mlp", False):
            mlp_proj = SimpleMLPProjection(
                input_dim=cfg.model.hidden_dim,
                output_dim=cfg.model.clip_dim,
            )
        else:
            mlp_proj = MLPProjection(
                input_dim=cfg.model.hidden_dim,
                output_dim=cfg.model.clip_dim,
                dropout=cfg.model.dropout,
            )

        discriminator = Discriminator(
            feature_dim=cfg.model.hidden_dim,
            global_dim=cfg.model.clip_dim,
        )

        text_encoder = CLIPTextEncoder(
            model_name=cfg.model.clip_model,
            device=device,
            use_learnable_prompts=use_learnable_prompts,
            n_ctx=getattr(cfg.model, "prompt_n_ctx", 16),
            ctx_init=getattr(cfg.model, "prompt_ctx_init", "a photo of a person"),
            num_classes=cfg.model.num_classes,
            learnable_temp=getattr(cfg.model, "learnable_temp", True),
        )

        global_rep = IntegratedGlobalRepresentation(
            num_classes=cfg.model.num_classes,
            feature_dim=cfg.model.clip_dim,
            rho=cfg.training.rho,
            warmup_epochs=cfg.training.warmup_epochs,
        )

        return cls(
            backbone=backbone,
            mlp_proj=mlp_proj,
            text_encoder=text_encoder,
            discriminator=discriminator,
            global_rep=global_rep,
            num_classes=cfg.model.num_classes,
            hidden_dim=cfg.model.hidden_dim,
            clip_dim=cfg.model.clip_dim,
            label_names=label_names,
            clip_subject=cfg.dataset.clip_subject,
            clip_template=cfg.dataset.clip_template,
            device=device,
            use_learnable_prompts=use_learnable_prompts,
            disable_geo_branch=disable_geo_branch,
            lambda4=getattr(cfg.training, "lambda4", 0.1),
        )

    # ---------------------------------------------------------------------------
    # Inițializare
    # ---------------------------------------------------------------------------

    def initialize_G(self, clip_visual_features: torch.Tensor, labels: torch.Tensor) -> None:
        proto = Prototyping(self.num_classes, self.clip_dim)
        g_init = proto(
            clip_visual_features.to(self.device),
            labels.to(self.device),
        )
        self.global_rep.initialize(g_init)
        print(f"  G initialized with CLIP visual prototypes (shape: {g_init.shape})")

    def initialize_G_from_text(self) -> None:
        self.global_rep.initialize(self.T.clone())
        print(f"  G initialized from text features T (shape: {self.T.shape})")

    # ---------------------------------------------------------------------------
    # Forward pass
    # ---------------------------------------------------------------------------

    def forward(
        self,
        roi_features: torch.Tensor,
        geo_features: Optional[torch.Tensor] = None,
        entity_types: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        training_stage: int = 2,
    ) -> Dict[str, torch.Tensor]:
        if self.disable_geo_branch:
            geo_features = None

        backbone_out = self.backbone(
            roi_features=roi_features,
            geo_features=geo_features,
            entity_types=entity_types,
            training_stage=training_stage,
        )
        z              = backbone_out["z"]
        frame_logits   = backbone_out["frame_logits"]
        segment_logits = backbone_out["segment_logits"]
        u_soft         = backbone_out["u_soft"]
        segment_logits_bms = backbone_out.get("segment_logits_bms")
        u_soft_bms     = backbone_out.get("u_soft_bms")

        if self._inference_mode:
            return {
                "segment_logits": segment_logits,
                "frame_logits":   frame_logits,
            }

        z_prime = self.mlp_proj(z)

        G = self.global_rep.get_G()
        mi_scores = self.discriminator(z, G)

        if self.use_learnable_prompts:
            T = self.text_encoder.encode_labels(
                self._label_names,
                subject=self._clip_subject,
            )
            with torch.no_grad():
                self.T.copy_(T.detach())
        else:
            T = self.T

        cos_similarities = torch.einsum("bnd,cd->bnc", z_prime, T) * self.text_encoder.temperature

        if self.use_learnable_prompts:
            frozen_T = self.text_encoder.get_frozen_T()
            if frozen_T is not None:
                cos_sim_prompt = (T * frozen_T.to(T.device)).sum(dim=-1)
                prompt_reg_loss = (1.0 - cos_sim_prompt).mean()
            else:
                prompt_reg_loss = torch.tensor(0.0, device=z.device)
        else:
            prompt_reg_loss = torch.tensor(0.0, device=z.device)

        if labels is not None:
            self.collector.add(z_prime, labels)

        return {
            "segment_logits":     segment_logits,
            "frame_logits":       frame_logits,
            "u_soft":             u_soft,
            "segment_logits_bms": segment_logits_bms,
            "u_soft_bms":         u_soft_bms,
            "mi_scores":          mi_scores,
            "cos_similarities":   cos_similarities,
            "z_prime":            z_prime,
            "prompt_reg_loss":    prompt_reg_loss,
        }

    # ---------------------------------------------------------------------------
    # Utilitare training
    # ---------------------------------------------------------------------------

    def end_of_epoch(self, epoch: int) -> None:
        features, labels = self.collector.get_all()
        if features is not None:
            self.global_rep.update(
                features.to(self.device),
                labels.to(self.device),
                epoch,
            )
        self.collector.reset()

    def set_inference_mode(self, inference: bool = True) -> None:
        if inference and self.use_learnable_prompts:
            with torch.no_grad():
                T_final = self.text_encoder.encode_labels(
                    self._label_names,
                    subject=self._clip_subject,
                )
                self.T.copy_(T_final)

        self._inference_mode = inference
        if inference:
            self.eval()
        else:
            self.train()

    def get_trainable_params(self) -> tuple:
        trainable, frozen = [], []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable.append(name)
            else:
                frozen.append(name)
        return trainable, frozen

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total": total,
            "trainable": trainable,
            "frozen": total - trainable,
        }
