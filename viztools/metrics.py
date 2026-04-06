from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from viztools.composite import FrameBundle

try:
    import lpips as _lpips
except ImportError:  # pragma: no cover
    _lpips = None


def make_lpips_model(net: str = "alex", device: Optional[torch.device] = None):
    """AlexNet backbone LPIPS（与 perceptualsimilarity / Zhang et al. 一致）。"""
    if _lpips is None:
        raise RuntimeError("lpips is not installed")
    # lpips 仍调用 torchvision.alexnet(pretrained=...)，会触发 weights API 弃用提示（见 _utils）
    with warnings.catch_warnings():
        for msg in (
            r"The parameter 'pretrained' is deprecated",
            r"Arguments other than a weight enum or `None` for 'weights' are deprecated",
        ):
            warnings.filterwarnings("ignore", message=msg, category=UserWarning)
        m = _lpips.LPIPS(net=net, verbose=False)
    if device is not None:
        m = m.to(device)
    return m.eval()


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a - b) ** 2))


def psnr(a: np.ndarray, b: np.ndarray, data_range: float = 1.0) -> float:
    return float(peak_signal_noise_ratio(a, b, data_range=data_range))


def ssim_value(a: np.ndarray, b: np.ndarray) -> float:
    return float(
        structural_similarity(
            a,
            b,
            channel_axis=2,
            data_range=1.0,
        )
    )


def ssim_maps(
    a: np.ndarray, b: np.ndarray
) -> Tuple[np.ndarray, float]:
    """Returns per-channel SSIM map (mean over channels) and scalar SSIM."""
    ch_maps = []
    ssum = 0.0
    for c in range(3):
        m, full = structural_similarity(
            a[..., c],
            b[..., c],
            full=True,
            data_range=1.0,
        )
        ch_maps.append(full.astype(np.float32))
        ssum += m
    smap = np.mean(np.stack(ch_maps, axis=-1), axis=-1)
    return smap, ssum / 3.0


def abs_diff_heatmap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Single-channel heatmap in [0,1]: mean abs diff across RGB."""
    return np.mean(np.abs(a - b), axis=-1).astype(np.float32)


@dataclass
class PairMetrics:
    mse: float
    ssim: float
    psnr: float
    lpips: float


def lpips_pair(
    loss_fn: torch.nn.Module,
    a: np.ndarray,
    b: np.ndarray,
    device: torch.device,
) -> float:
    """a,b float [H,W,3] in [0,1]."""
    t_a = (
        torch.from_numpy(a)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
        * 2.0
        - 1.0
    )
    t_b = (
        torch.from_numpy(b)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
        .to(device)
        * 2.0
        - 1.0
    )
    with torch.no_grad():
        d = loss_fn(t_a, t_b)
    return float(d.item())


def compute_pair_metrics(
    a: np.ndarray,
    b: np.ndarray,
    loss_fn: Optional[torch.nn.Module],
    device: torch.device,
    need_lpips: bool,
) -> PairMetrics:
    m = mse(a, b)
    ssim_v = ssim_value(a, b)
    p = psnr(a, b)
    lp = 0.0
    if need_lpips and loss_fn is not None:
        lp = lpips_pair(loss_fn, a, b, device)
    return PairMetrics(mse=m, ssim=ssim_v, psnr=p, lpips=lp)


def batch_lpips_pairs(
    loss_fn: torch.nn.Module,
    device: torch.device,
    pairs: List[Tuple[np.ndarray, np.ndarray]],
    chunk: int = 24,
) -> List[float]:
    """Batch LPIPS for many pairs on GPU for speed（分块避免显存峰值）。"""
    if not pairs:
        return []
    out: List[float] = []
    for start in range(0, len(pairs), chunk):
        sub = pairs[start : start + chunk]
        xs = [torch.from_numpy(a).permute(2, 0, 1).float() for a, _ in sub]
        ys = [torch.from_numpy(b).permute(2, 0, 1).float() for _, b in sub]
        bx = torch.stack(xs, dim=0).to(device) * 2.0 - 1.0
        by = torch.stack(ys, dim=0).to(device) * 2.0 - 1.0
        with torch.no_grad():
            d = loss_fn(bx, by).view(-1)
        out.extend(float(x) for x in d.cpu().tolist())
    return out


def compute_metrics_table(
    bundles: List[FrameBundle],
    loss_fn: Optional[torch.nn.Module],
    device: torch.device,
    need_lpips: bool,
) -> List[Dict[str, float]]:
    """All frames; LPIPS batched per forward when possible."""
    rows: List[Dict[str, float]] = []
    if not bundles:
        return rows

    # numpy metrics first
    lp_ir: List[Tuple[np.ndarray, np.ndarray]] = []
    lp_ig: List[Tuple[np.ndarray, np.ndarray]] = []
    lp_rg: List[Tuple[np.ndarray, np.ndarray]] = []

    for b in bundles:
        inp, recon, gt = b.input_rgb, b.reconstruction_rgb, b.gt_rgb
        pm_ir = compute_pair_metrics(inp, recon, None, device, need_lpips=False)
        pm_ig = compute_pair_metrics(inp, gt, None, device, need_lpips=False)
        pm_rg = compute_pair_metrics(recon, gt, None, device, need_lpips=False)
        row = {
            "frame_id": float(b.frame_id),
            "ir_mse": pm_ir.mse,
            "ir_ssim": pm_ir.ssim,
            "ir_psnr": pm_ir.psnr,
            "ig_mse": pm_ig.mse,
            "ig_ssim": pm_ig.ssim,
            "ig_psnr": pm_ig.psnr,
            "rg_mse": pm_rg.mse,
            "rg_ssim": pm_rg.ssim,
            "rg_psnr": pm_rg.psnr,
        }
        rows.append(row)
        if need_lpips and loss_fn is not None:
            lp_ir.append((inp, recon))
            lp_ig.append((inp, gt))
            lp_rg.append((recon, gt))

    if need_lpips and loss_fn is not None and bundles:
        ir_l = batch_lpips_pairs(loss_fn, device, lp_ir)
        ig_l = batch_lpips_pairs(loss_fn, device, lp_ig)
        rg_l = batch_lpips_pairs(loss_fn, device, lp_rg)
        for i, row in enumerate(rows):
            row["ir_lpips"] = ir_l[i]
            row["ig_lpips"] = ig_l[i]
            row["rg_lpips"] = rg_l[i]
    else:
        for row in rows:
            row["ir_lpips"] = 0.0
            row["ig_lpips"] = 0.0
            row["rg_lpips"] = 0.0

    return rows
