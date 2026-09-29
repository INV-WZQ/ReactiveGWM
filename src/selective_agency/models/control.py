"""Complete Action/NPC text groups for Actor-bound control attention."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _require_integer(value: torch.Tensor, name: str) -> None:
    if value.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
        raise ValueError(f"{name} must contain integer indices")


class ControlTokenBuilder(nn.Module):
    """Assemble variable-length controls and one background sentinel.

    ``control_kind`` is 0 for padding, 1 for external Action text, and 2 for an
    NPC's complete conditional prompt. Unused indices are never read. Raw T5
    features may be cached, but their trainable projection is evaluated here on
    every forward, once per distinct referenced text row.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        del eps
        self.dim = int(dim)
        self.background_sentinel = nn.Parameter(torch.zeros(1, self.dim))

    def _project_used_text(
        self,
        text_embedding: nn.Module,
        hidden: torch.Tensor | None,
        mask: torch.Tensor | None,
        ids: torch.Tensor,
        active: torch.Tensor,
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Project the batch's unique rows, removing only trailing padding."""

        weight = next(text_embedding.parameters())
        used_ids = ids[active]
        if not used_ids.numel():
            shape = (*ids.shape, 0)
            return (
                torch.zeros(*shape, self.dim, device=ids.device, dtype=weight.dtype),
                torch.zeros(shape, device=ids.device, dtype=torch.bool),
            )
        if hidden is None or mask is None or hidden.ndim != 3 or hidden.shape[0] == 0:
            raise ValueError(f"{name} text cache is required by active controls")
        if mask.shape != hidden.shape[:2]:
            raise ValueError(f"{name} text mask shape disagrees with hidden states")
        if bool(((used_ids < 0) | (used_ids >= hidden.shape[0])).any()):
            raise ValueError(f"{name} id outside configured text vocabulary")

        unique_ids = torch.unique(used_ids)
        used_mask = mask.index_select(0, unique_ids.to(mask.device)).to(
            device=ids.device, dtype=torch.bool
        )
        if not bool(used_mask.any(dim=-1).all()):
            raise ValueError(f"{name} text rows must contain at least one valid token")
        # A last-valid position (rather than the mask sum) also preserves caches
        # with internal masked positions. There is no Action or NPC length cap.
        length = int(used_mask.any(dim=0).nonzero()[-1, 0]) + 1
        used_mask = used_mask[:, :length]
        used_hidden = hidden.index_select(0, unique_ids.to(hidden.device))[:, :length]
        used_hidden = used_hidden.to(device=ids.device, dtype=weight.dtype)
        projected = text_embedding(used_hidden.masked_fill(~used_mask[..., None], 0))

        # The filler refers to an actual used row, even when row zero is unused.
        # Thus -1 and arbitrary unused fields can never become PyTorch indices.
        safe_ids = torch.where(active, ids, unique_ids[0])
        inverse = torch.searchsorted(unique_ids, safe_ids)
        valid = used_mask[inverse] & active[..., None]
        return projected[inverse].masked_fill(~valid[..., None], 0), valid

    def forward(
        self,
        *,
        text_embedding: nn.Module,
        control_text_hidden: torch.Tensor,
        control_text_mask: torch.Tensor,
        action_ids: torch.Tensor,
        subject_valid: torch.Tensor,
        control_kind: torch.Tensor | None = None,
        npc_prompt_ids: torch.Tensor | None = None,
        npc_text_hidden: torch.Tensor | None = None,
        npc_text_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return tokens ``[B,T,1+S,L,D]`` and validity ``[B,T,1+S,L]``.

        ``L`` depends only on the text rows referenced by this batch, never the
        padded width of the cache. Each NPC group contains its complete prompt;
        the attention layer binds that group to its Actor handle through Key.
        """

        if action_ids.ndim != 3 or not action_ids.numel():
            raise ValueError("action ids must be nonempty [B,T,S]")
        _require_integer(action_ids, "action ids")
        batch, steps, subject_count = action_ids.shape
        if subject_valid.shape != (batch, subject_count):
            raise ValueError("subject validity shape disagrees with action ids")
        subject_valid = subject_valid.to(device=action_ids.device, dtype=torch.bool)
        if control_kind is None:
            control_kind = subject_valid.long()
        if control_kind.shape != subject_valid.shape:
            raise ValueError("control kind shape disagrees with subject validity")
        _require_integer(control_kind, "control kind")
        control_kind = control_kind.to(action_ids.device)
        if bool(((control_kind < 0) | (control_kind > 2)).any()):
            raise ValueError("control kind must be padding=0, external=1, or NPC=2")
        if not torch.equal(control_kind != 0, subject_valid):
            raise ValueError("control kind must agree with subject validity")
        external = control_kind == 1
        npc = control_kind == 2

        if npc_prompt_ids is None:
            npc_prompt_ids = torch.full_like(control_kind, -1)
        if npc_prompt_ids.shape != subject_valid.shape:
            raise ValueError("NPC prompt ids must be [B,S]")
        _require_integer(npc_prompt_ids, "NPC prompt ids")
        npc_prompt_ids = npc_prompt_ids.to(device=action_ids.device, dtype=torch.long)

        action_tokens, action_valid = self._project_used_text(
            text_embedding,
            control_text_hidden,
            control_text_mask,
            action_ids.long(),
            external[:, None].expand(-1, steps, -1),
            "Action",
        )
        npc_tokens, npc_valid = self._project_used_text(
            text_embedding,
            npc_text_hidden,
            npc_text_mask,
            npc_prompt_ids,
            npc,
            "NPC prompt",
        )
        action_length, npc_length = action_tokens.shape[-2], npc_tokens.shape[-2]
        length = max(action_length, npc_length, 1)
        subject_tokens = F.pad(action_tokens, (0, 0, 0, length - action_length))
        subject_tokens = subject_tokens + F.pad(npc_tokens, (0, 0, 0, length - npc_length))[:, None]
        column_valid = (
            F.pad(action_valid, (0, length - action_length))
            | F.pad(npc_valid, (0, length - npc_length))[:, None]
        )

        background_tokens = F.pad(
            self.background_sentinel.to(subject_tokens.dtype), (0, 0, 0, length - 1)
        )[None, None, None].expand(batch, steps, 1, -1, -1)
        background_valid = torch.zeros(
            batch, steps, 1, length, dtype=torch.bool, device=action_ids.device
        )
        background_valid[..., 0] = True
        return (
            torch.cat((background_tokens, subject_tokens), dim=2),
            torch.cat((background_valid, column_valid), dim=2),
        )
