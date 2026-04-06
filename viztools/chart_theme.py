"""与 Streamlit 当前主题同步的 Plotly 样式（高对比度文字与网格）。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go


@dataclass(frozen=True)
class ChartTheme:
    plotly_template: str
    paper_bg: str
    plot_bg: str
    text_primary: str
    text_secondary: str
    grid: str
    axis_line: str


def _color_scheme_from_streamlit() -> str:
    """Streamlit ≥1.46 使用 `theme.type`（由浏览器 color_scheme 推断），旧版可能只有 `base`。"""
    try:
        import streamlit as st

        th: Any = st.context.theme
        # 正确属性名是 type，不是 base（base 不存在时会误用默认值恒为 light）
        typ = getattr(th, "type", None) or getattr(th, "base", None)
        if typ in ("dark", "light"):
            return typ
    except Exception:
        pass
    return "light"


def chart_theme_from_streamlit() -> ChartTheme:
    """读取 `st.context.theme.type`，与 Streamlit 设置里的明暗模式一致。

    说明：`StreamlitTheme` 只保证提供 `type`，不包含 CSS 里的 backgroundColor；
    这里用与默认暗色/亮色应用相近的固定色。若切换主题后未立即更新，见
    https://github.com/streamlit/streamlit/issues/11920（可手动刷新页面触发重算）。
    """
    scheme = _color_scheme_from_streamlit()
    is_dark = scheme == "dark"

    if is_dark:
        return ChartTheme(
            plotly_template="plotly_dark",
            paper_bg="#0e1117",
            plot_bg="#0e1117",
            text_primary="#f8fafc",
            text_secondary="#e2e8f0",
            grid="#64748b",
            axis_line="#94a3b8",
        )
    return ChartTheme(
        plotly_template="plotly_white",
        paper_bg="#ffffff",
        plot_bg="#ffffff",
        text_primary="#0f172a",
        text_secondary="#334155",
        grid="#cbd5e1",
        axis_line="#64748b",
    )


def apply_chart_theme(
    fig: go.Figure,
    theme: ChartTheme,
    *,
    axes: bool = True,
) -> go.Figure:
    """统一字体、坐标轴、子图标题、图例与热力图 colorbar 对比度。

    `axes=False` 用于 `go.Image` 等仅保留背景色、不覆盖网格/坐标轴样式的图。
    """
    fig.update_layout(
        template=theme.plotly_template,
        paper_bgcolor=theme.paper_bg,
        plot_bgcolor=theme.plot_bg,
        font=dict(
            family="Inter, 'Segoe UI', system-ui, sans-serif",
            size=13,
            color=theme.text_primary,
        ),
        title_font=dict(size=15, color=theme.text_primary),
        hoverlabel=dict(
            font=dict(size=12, color=theme.text_primary),
            bgcolor=theme.plot_bg,
            bordercolor=theme.grid,
        ),
    )
    fig.update_annotations(
        font=dict(size=13, color=theme.text_primary),
    )
    if axes:
        axis_kw = dict(
            gridcolor=theme.grid,
            gridwidth=1,
            showgrid=True,
            zeroline=False,
            linecolor=theme.axis_line,
            linewidth=1,
            mirror=True,
            tickfont=dict(size=11, color=theme.text_secondary),
            title_font=dict(size=12, color=theme.text_primary),
        )
        fig.update_xaxes(**axis_kw)
        fig.update_yaxes(**axis_kw)

    fig.update_traces(
        selector=dict(type="heatmap"),
        colorbar=dict(
            tickfont=dict(size=11, color=theme.text_secondary),
            title_font=dict(size=12, color=theme.text_primary),
            outlinecolor=theme.axis_line,
            outlinewidth=1,
        ),
    )
    return fig
