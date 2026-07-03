"""
models/base.py
Interfețe și clase abstracte pentru HOLIC.

Scop: expun contractele principale ale arhitecturii astfel încât
orchestratorul HOLIC să depindă de abstracțiuni, nu de implementări concrete.
Aceasta permite:
  - Moștenire / implementare în diagrama UML
  - Injecție de dependențe (aggregare în loc de compoziție exclusivă)
  - Testare unitară cu mock-uri
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
import torch
import torch.nn as nn


class IBackbone(nn.Module, ABC):
    """Contract pentru backbone-ul spatio-temporal (2G-GCN + ASSIGN)."""

    @abstractmethod
    def forward(
        self,
        roi_features: torch.Tensor,
        geo_features: Optional[torch.Tensor] = None,
        entity_types: Optional[torch.Tensor] = None,
        training_stage: int = 2,
    ) -> Dict[str, torch.Tensor]:
        """
        Returnează dict cu:
          z, frame_logits, segment_logits, u_soft, u_hard,
          segment_logits_bms, u_soft_bms, u_hard_bms
        """
        ...

    @abstractmethod
    def set_gsm_temperature(self, temp: float) -> None:
        """Ajustează temperatura Gumbel-Softmax a detectorului de granite."""
        ...


class IProjection(nn.Module, ABC):
    """Contract pentru proiecția Z -> Z' (spatiul CLIP)."""

    @abstractmethod
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        ...


class ITextEncoder(nn.Module, ABC):
    """Contract pentru encoder text (CLIP static sau learnable prompts C6b)."""

    @property
    @abstractmethod
    def temperature(self) -> torch.Tensor:
        """Temperatură scalabilă pentru cosine similarities."""
        ...

    @abstractmethod
    def encode_labels(
        self,
        label_names: List[str],
        subject: str = "person",
        template=None,
    ) -> torch.Tensor:
        """Returnează T: (C, clip_dim) — L2 normalizat."""
        ...

    @abstractmethod
    def precompute_frozen_T(
        self,
        label_names: List[str],
        subject: str,
        template,
    ) -> torch.Tensor:
        """Calculează și memorează T_frozen (fără gradient)."""
        ...

    @abstractmethod
    def get_frozen_T(self) -> Optional[torch.Tensor]:
        """Returnează T_frozen dacă a fost precomputat."""
        ...
