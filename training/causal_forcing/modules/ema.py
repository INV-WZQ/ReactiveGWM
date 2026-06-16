"""轻量 EMA — Stage 2 CD / Stage 3 DMD 共用.

对齐 CF 上游 ``utils/distributed.py::EMA_FSDP``: **master 累加器是 fp32 shadow**
(可放 CPU 或 GPU), 每步 fp32 累加; 另有一个常驻 GPU 的 **bf16 model 副本** 作为
"目标网络" forward 载体 (CD 每步要 forward 出 ``cm_pred_t_next``), 每步从 shadow
``copy_to`` 同步 (cast 到 bf16).

为什么不直接把 EMA 存 bf16: decay=0.99 时每步增量 ≈ ``0.01·(p−e)``, 若小于 bf16 的
ULP (相对精度 ~0.4%) 会被舍入吃掉 → 长程 (50K 步) EMA 跟不动 / 系统性滞后. fp32
shadow 累加避免此问题 (官方做法). 最终导出/forward 时才 cast bf16.

- ``decay=0.99`` **从 step 0 就更新** (PLAN §2 Stage 2); ``ema_start_step`` 仅决定
  导出时存 EMA 还是 raw generator, **不是** "200 步才开 EMA".
- 只覆盖 ``requires_grad`` 对应的参数名 (buffers 如 RoPE freqs 是常量, 不动).
- ``src_model`` 应为 **unwrap 后的 generator** (``accelerator.unwrap_model(model).pipe.dit``),
  其 ``named_parameters()`` 与 shadow / EMA model 的 key 一一对应.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def ema_init_shadow(
    src_model: torch.nn.Module, device: str | torch.device = "cpu"
) -> dict[str, torch.Tensor]:
    """从 ``src_model`` 建 fp32 master shadow (官方 ``EMA_FSDP._init_shadow``).

    默认放 CPU (不占 GPU 显存; 3×5B 同驻时关键). 设 ``device="cuda"`` 可放 GPU 换取
    速度 (省去每步 CPU↔GPU 传输), 但 +约 2× param 字节的 VRAM.
    """
    return {
        n: p.detach().float().clone().to(device)
        for n, p in src_model.named_parameters()
    }


@torch.no_grad()
def ema_update_shadow(
    shadow: dict[str, torch.Tensor], src_model: torch.nn.Module, decay: float
) -> None:
    """``shadow = decay·shadow + (1−decay)·src``, **全程 fp32** (官方 ``EMA_FSDP.update``).

    在 shadow 所在 device 上做累加; src param 先 ``.float().to(shadow.device)``.
    """
    for n, p in src_model.named_parameters():
        s = shadow.get(n)
        if s is None:
            continue
        s.mul_(decay).add_(p.detach().float().to(s.device), alpha=1.0 - decay)


@torch.no_grad()
def ema_copy_to_model(
    shadow: dict[str, torch.Tensor], ema_model: torch.nn.Module
) -> None:
    """fp32 shadow → ``ema_model`` 参数 (cast 到 model dtype, 供 forward; 官方 ``copy_to``)."""
    for n, p in ema_model.named_parameters():
        s = shadow.get(n)
        if s is not None:
            p.copy_(s.to(dtype=p.dtype, device=p.device))
