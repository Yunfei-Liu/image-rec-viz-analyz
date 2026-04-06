from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from io import BytesIO
from PIL import Image

_FRAME_RE = re.compile(r"^(\d+)\.(png|jpg|jpeg|webp)$", re.IGNORECASE)


@dataclass(frozen=True)
class FrameBundle:
    """One video frame: four horizontal panels — input, reference, reconstruction, gt."""

    frame_id: int
    path: Path
    input_rgb: np.ndarray  # float32 [H,W,3] in [0,1]
    reference_rgb: np.ndarray
    reconstruction_rgb: np.ndarray
    gt_rgb: np.ndarray


def _to_float_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.float32 and arr.max() <= 1.0:
        return np.clip(arr, 0.0, 1.0)
    if arr.dtype == np.uint8:
        return (arr.astype(np.float32) / 255.0).clip(0.0, 1.0)
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def split_four_panels(
    rgb: np.ndarray, n_panels: int = 4
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split a composite image into four equal-width strips (left → right)."""
    h, w = rgb.shape[:2]
    if w % n_panels != 0:
        # trim right edge so split is exact
        w = (w // n_panels) * n_panels
        rgb = rgb[:, :w, :]
    cw = w // n_panels
    panels = [rgb[:, i * cw : (i + 1) * cw, :] for i in range(n_panels)]
    return panels[0], panels[1], panels[2], panels[3]


def load_frame_bundle_from_bytes(filename: str, data: bytes, n_panels: int = 4) -> FrameBundle:
    stem = Path(filename).stem
    if not stem.isdigit():
        raise ValueError(f"Expected numeric frame id in filename, got: {filename}")
    frame_id = int(stem)
    im = Image.open(BytesIO(data)).convert("RGB")
    arr = np.asarray(im)
    rgb = _to_float_rgb(arr)
    inp, ref, recon, gt = split_four_panels(rgb, n_panels=n_panels)
    return FrameBundle(
        frame_id=frame_id,
        path=Path(filename),
        input_rgb=inp,
        reference_rgb=ref,
        reconstruction_rgb=recon,
        gt_rgb=gt,
    )


def load_frame_bundle(path: Path | str, n_panels: int = 4) -> FrameBundle:
    path = Path(path)
    stem = path.stem
    if not stem.isdigit():
        raise ValueError(f"Expected numeric frame id in filename, got: {path.name}")
    frame_id = int(stem)
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im)
    rgb = _to_float_rgb(arr)
    inp, ref, recon, gt = split_four_panels(rgb, n_panels=n_panels)
    return FrameBundle(
        frame_id=frame_id,
        path=path,
        input_rgb=inp,
        reference_rgb=ref,
        reconstruction_rgb=recon,
        gt_rgb=gt,
    )


def list_frame_paths(folder: Path | str, extensions: Sequence[str] | None = None) -> List[Path]:
    folder = Path(folder)
    if not folder.is_dir():
        raise NotADirectoryError(str(folder))
    exts = {e.lower().lstrip(".") for e in (extensions or ("png", "jpg", "jpeg", "webp"))}
    paths: List[Path] = []
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        m = _FRAME_RE.match(p.name)
        if not m:
            continue
        if m.group(2).lower() not in exts:
            continue
        paths.append(p)
    paths.sort(key=lambda x: int(x.stem))
    return paths


def iter_bundles(paths: Iterable[Path], n_panels: int = 4) -> Iterable[FrameBundle]:
    for p in paths:
        yield load_frame_bundle(p, n_panels=n_panels)
