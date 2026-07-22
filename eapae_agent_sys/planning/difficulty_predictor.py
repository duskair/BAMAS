import os
import re
import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Union, List, Literal
from transformers import AutoConfig, AutoModel, AutoTokenizer

# Define the DeviceLike type matching AgentBalance
DeviceLike = Union[str, torch.device]
_MODEL_DIR = Path('outputs/checkpoints/')

class DifficultyPredictor2(nn.Module):
    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        device: DeviceLike = "cpu",
        freeze_encoder: bool = True,
        prepend_template: bool = False,
        load_path: Optional[Union[str, Path]] = None,
        model_src: str = "sentence-transformers/all-mpnet-base-v2",
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.device = torch.device(device)
        self.freeze_encoder = freeze_encoder
        self.prepend_template = prepend_template
        self.model_src = model_src
        self.local_files_only = local_files_only

        self._init_hidden_dim = hidden_dim
        self._init_dropout = dropout

        self.tokenizer: Optional[AutoTokenizer] = None
        self.encoder: Optional[AutoModel] = None
        self.reg_head: Optional[nn.Sequential] = None
        self._pending_state_dict: Optional[dict] = None
        self._built: bool = False

        if load_path is not None:
            ckpt_path = Path(load_path)
            sd = torch.load(ckpt_path, map_location=self.device)
            if not isinstance(sd, dict):
                raise RuntimeError(f"Checkpoint at {ckpt_path} is not a state_dict.")
            self._pending_state_dict = sd
            
        self._ensure_built()
        self._ensure_tokenizer()

    def _ensure_built(self) -> None:
        if self._built:
            return

        cfg = AutoConfig.from_pretrained(self.model_src, local_files_only=self.local_files_only)
        self.encoder = AutoModel.from_config(cfg).to(self.device)

        hidden_size = cfg.hidden_size
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_size, self._init_hidden_dim),
            nn.GELU(),
            nn.Dropout(self._init_dropout),
            nn.Linear(self._init_hidden_dim, 1),
        ).to(self.device)

        self.tokenizer = None

        if self._pending_state_dict is not None:
            self.load_state_dict(self._pending_state_dict, strict=True)
            self._pending_state_dict = None

        if self.freeze_encoder and self.encoder is not None:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad_(False)

        self._built = True

    def _ensure_tokenizer(self):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_src, local_files_only=self.local_files_only
            )

    def _encode_batch(self, texts: List[str]) -> torch.Tensor:
        self._ensure_built()
        self._ensure_tokenizer()
        toks = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
        out = self.encoder(**toks).last_hidden_state
        mask = toks.attention_mask.unsqueeze(-1)
        return (out * mask).sum(1) / mask.sum(1)
    
    @staticmethod
    def _apply_template(texts: List[str]) -> List[str]:
        out = []
        for t in texts:
            s = "" if t is None else str(t).strip()
            s_lower = s.lower()
            if s_lower.startswith("intent:") or s.startswith("[Intent]"):
                out.append(s)
            else:
                out.append(
                    "Intent: Judge the task difficulty as a continuous score in [0, 1] "
                    "(0 = very easy, 1 = very hard). Base your judgment only on the prompt.\n"
                    "Question:\n" + s
                )
        return out

    def encode(self, texts: List[str], batch_size: int = 32) -> torch.Tensor:
        vecs = []
        rng = range(0, len(texts), batch_size)
        if self.freeze_encoder:
            with torch.no_grad():
                for i in rng:
                    vecs.append(self._encode_batch(texts[i:i+batch_size]))
        else:
            for i in rng:
                vecs.append(self._encode_batch(texts[i:i+batch_size]))
        return torch.cat(vecs, 0)

    def forward(self, texts: List[str], *, apply_sigmoid: bool = False) -> torch.Tensor:
        if self.prepend_template:
            texts = self._apply_template(texts)
        emb = self.encode(texts)
        logits = self.reg_head(emb).squeeze(-1)
        return torch.sigmoid(logits) if apply_sigmoid else logits
    
    def save_full(self, path: Union[str, Path] | None = None) -> None:
        # Defaults to BAMAS outputs/checkpoints/ directory if no path is provided
        _MODEL_DIR.mkdir(parents=True, exist_ok=True)
        path = Path(path or _MODEL_DIR / "difficulty_predictor2_full.pt")
        torch.save(self.state_dict(), path)
        print("[DifficultyPredictor2] full model saved →", path)

    def change_device(self, device: DeviceLike) -> None:
        self.device = torch.device(device)
        self.encoder.to(self.device)
        self.reg_head.to(self.device)