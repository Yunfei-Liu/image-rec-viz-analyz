from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from viztools.chart_theme import ChartTheme, apply_chart_theme


def metrics_figure(
    df: pd.DataFrame, theme: ChartTheme, *, chart_height: int = 900
) -> go.Figure:
    """Four rows: MSE, SSIM, PSNR, LPIPS. Three lines per subplot."""
    if df.empty:
        fig = go.Figure()
        fig.update_layout(title=dict(text="无数据 — 请先加载结果目录", x=0.5))
        apply_chart_theme(fig, theme)
        fig.update_layout(height=480)
        return fig

    x = df["frame_id"].astype(int)
    pairs = [
        ("ir", "Input–Recon", "#636efa"),
        ("ig", "Input–GT", "#ef553b"),
        ("rg", "Recon–GT", "#00cc96"),
    ]
    metrics = [
        ("mse", "MSE", "MSE (↓)"),
        ("ssim", "SSIM", "SSIM (↑)"),
        ("psnr", "PSNR", "PSNR (↑) dB"),
        ("lpips", "LPIPS", "LPIPS (↓)"),
    ]

    fig = make_subplots(
        rows=4,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[m[2] for m in metrics],
    )

    for row, (suffix, _short, _title) in enumerate(metrics, start=1):
        for key, name, color in pairs:
            col = f"{key}_{suffix}"
            if col not in df.columns:
                continue
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=df[col],
                    mode="lines+markers",
                    name=name if row == 1 else name,
                    legendgroup=key,
                    showlegend=(row == 1),
                    line=dict(width=2, color=color),
                    marker=dict(size=6, line=dict(width=0)),
                    hovertemplate=f"{name}<br>frame=%{{x}}<br>{suffix}=%{{y:.6f}}<extra></extra>",
                ),
                row=row,
                col=1,
            )

    fig.update_xaxes(title_text="Frame ID", row=4, col=1)
    apply_chart_theme(fig, theme)
    fig.update_layout(
        height=chart_height,
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12, color=theme.text_primary),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=48, r=24, t=48, b=48),
    )
    return fig


def heatmap_figure(
    h1: np.ndarray,
    h2: np.ndarray,
    title1: str,
    title2: str,
    theme: ChartTheme,
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(title1, title2),
        horizontal_spacing=0.06,
    )
    for i, (h, title) in enumerate([(h1, title1), (h2, title2)], start=1):
        fig.add_trace(
            go.Heatmap(
                z=np.flipud(h),
                colorscale="Turbo",
                showscale=True,
                hovertemplate="x=%{x}<br>y=%{y}<br>z=%{z:.4f}<extra></extra>",
            ),
            row=1,
            col=i,
        )
    apply_chart_theme(fig, theme)
    fig.update_layout(height=360, margin=dict(l=12, r=12, t=56, b=12))
    return fig


def ssim_maps_figure(
    s1: np.ndarray, s2: np.ndarray, t1: str, t2: str, theme: ChartTheme
) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(t1, t2),
        horizontal_spacing=0.06,
    )
    for i, s in enumerate([s1, s2], start=1):
        z = np.clip(s, 0.0, 1.0)
        fig.add_trace(
            go.Heatmap(
                z=np.flipud(z),
                colorscale="RdYlGn",
                zmin=0,
                zmax=1,
                showscale=True,
                hovertemplate="x=%{x}<br>y=%{y}<br>ssim=%{z:.4f}<extra></extra>",
            ),
            row=1,
            col=i,
        )
    apply_chart_theme(fig, theme)
    fig.update_layout(height=360, margin=dict(l=12, r=12, t=56, b=12))
    return fig


def panel_strip_figure(
    inp: np.ndarray,
    ref: np.ndarray,
    recon: np.ndarray,
    gt: np.ndarray,
    theme: ChartTheme,
) -> go.Figure:
    """Single row RGB panels for quick preview."""
    h, w, _ = inp.shape
    strip = np.concatenate([inp, ref, recon, gt], axis=1)
    strip_u8 = (np.clip(strip, 0, 1) * 255).astype(np.uint8)
    fig = go.Figure(
        data=go.Image(z=strip_u8),
    )
    fig.update_xaxes(showgrid=False, range=[0, strip_u8.shape[1]], constrain="domain")
    fig.update_yaxes(showgrid=False, range=[h, 0], scaleanchor="x", scaleratio=1)
    apply_chart_theme(fig, theme, axes=False)
    fig.update_layout(
        height=min(420, h + 40),
        margin=dict(l=0, r=0, t=36, b=0),
        title=dict(
            text="Input · Reference · Reconstruction · GT",
            x=0.5,
            xanchor="center",
            font=dict(size=14, color=theme.text_primary),
        ),
    )
    return fig
