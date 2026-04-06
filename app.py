"""
Streamlit 可视化：推理结果四宫格（Input | Reference | Reconstruction | GT）分析。

运行：cd viz-tools && pip install -r requirements.txt && streamlit run app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from viztools.composite import (
    FrameBundle,
    list_frame_paths,
    load_frame_bundle,
)
from viztools.metrics import (
    abs_diff_heatmap,
    compute_metrics_table,
    make_lpips_model,
    ssim_maps,
)
from viztools.chart_theme import chart_theme_from_streamlit
from viztools.plots import heatmap_figure, metrics_figure, panel_strip_figure, ssim_maps_figure


def _inject_css() -> None:
    st.markdown(
        """
        <style>
          /* 随「宽屏」布局占满主区域，仅保留左右内边距 */
          .block-container {
            padding-top: 1.2rem;
            max-width: none;
            padding-left: clamp(0.75rem, 2.5vw, 2.5rem);
            padding-right: clamp(0.75rem, 2.5vw, 2.5rem);
          }
          div[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def get_lpips_model():
    return make_lpips_model(device=get_device())


def build_dataframe(rows: List[Dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("frame_id").reset_index(drop=True)


def resolve_paths(
    folder: Optional[str], uploads: Optional[List[Tuple[str, bytes]]]
) -> Tuple[List[Path], str]:
    if uploads:
        td = Path(tempfile.mkdtemp(prefix="viztools_upload_"))
        for name, data in uploads:
            if not name.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
                continue
            stem = Path(name).stem
            if not stem.isdigit():
                continue
            (td / Path(name).name).write_bytes(data)
        paths = list_frame_paths(td)
        return paths, "上传文件"
    if folder and Path(folder).expanduser().is_dir():
        p = list_frame_paths(Path(folder).expanduser())
        return p, "目录路径"
    return [], ""


def run_analysis(
    paths: List[Path], n_panels: int, use_lpips: bool
) -> Tuple[pd.DataFrame, Dict[int, Path]]:
    if not paths:
        return pd.DataFrame(), {}
    id_to_path = {int(p.stem): p for p in paths}
    device = get_device()
    loss_fn = get_lpips_model() if use_lpips else None
    bundles = [load_frame_bundle(p, n_panels=n_panels) for p in paths]
    rows = compute_metrics_table(bundles, loss_fn, device, need_lpips=use_lpips)
    return build_dataframe(rows), id_to_path


def load_bundle_for_frame(
    frame_id: int, id_to_path: Dict[int, Path], n_panels: int
) -> Optional[FrameBundle]:
    p = id_to_path.get(frame_id)
    if p is None:
        return None
    return load_frame_bundle(p, n_panels=n_panels)


def _apply_plotly_selection(event: Any, frame_ids: List[int]) -> None:
    if event is None:
        return
    try:
        sel = event.get("selection") if isinstance(event, dict) else getattr(event, "selection", None)
        if sel is None:
            return
        pts = sel.get("points") if isinstance(sel, dict) else getattr(sel, "points", None)
        if not pts:
            return
        p0 = pts[0]
        x = p0.get("x") if isinstance(p0, dict) else getattr(p0, "x", None)
        if x is None:
            return
        clicked = int(round(float(x)))
        if clicked in frame_ids:
            st.session_state.selected_frame_id = clicked
    except (TypeError, ValueError, KeyError, AttributeError):
        return


def main() -> None:
    st.set_page_config(
        page_title="推理结果分析",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    if "selected_frame_id" not in st.session_state:
        st.session_state.selected_frame_id = None
    if "metrics_df" not in st.session_state:
        st.session_state.metrics_df = pd.DataFrame()
    if "id_to_path" not in st.session_state:
        st.session_state.id_to_path = {}

    st.title("推理结果可视化")
    st.caption("四宫格：Input · Reference · Reconstruction · GT — 差异热力图 · SSIM 图 · 全序列指标曲线")

    with st.sidebar:
        st.subheader("数据源")
        folder = st.text_input(
            "结果文件夹路径",
            placeholder="/path/to/frames",
            help="包含 00001.png 等命名帧的目录（按文件名数字排序）",
        )
        up = st.file_uploader(
            "或批量拖入图片",
            type=["png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
            help="可多选同一文件夹下的全部帧图（浏览器无法直接拖文件夹时可用此项）",
        )
        uploads: Optional[List[Tuple[str, bytes]]] = None
        if up:
            uploads = [(f.name, f.getvalue()) for f in up]

        n_panels = st.number_input("横向子图数量", min_value=2, max_value=8, value=4, step=1)
        use_lpips = st.toggle("计算 LPIPS（首次会下载权重，较慢）", value=True)
        if st.button("加载并计算指标", type="primary", use_container_width=True):
            paths, src = resolve_paths(folder.strip() or None, uploads)
            if not paths:
                st.error("未找到有效帧：请检查路径或上传以数字命名的 png/jpg。")
            else:
                with st.spinner("正在计算 MSE / SSIM / PSNR" + (" / LPIPS …" if use_lpips else " …")):
                    df, idmap = run_analysis(paths, int(n_panels), use_lpips)
                st.session_state.metrics_df = df
                st.session_state.id_to_path = {k: Path(v) for k, v in idmap.items()}
                st.session_state.n_panels = int(n_panels)
                st.session_state.source_label = src
                if not df.empty:
                    st.session_state.selected_frame_id = int(df["frame_id"].iloc[0])
                st.success(f"已加载 {len(paths)} 帧（{src}）")
                st.rerun()

        st.divider()
        st.caption("设备: **%s**" % ("CUDA" if torch.cuda.is_available() else "CPU"))

    df: pd.DataFrame = st.session_state.metrics_df
    id_to_path: Dict[int, Path] = st.session_state.id_to_path
    n_panels = int(st.session_state.get("n_panels", 4))

    if df.empty:
        st.info("请在左侧输入目录路径或上传帧图后，点击「加载并计算指标」。")
        return

    frame_ids = [int(x) for x in df["frame_id"].tolist()]
    sel = st.session_state.selected_frame_id
    if sel not in frame_ids:
        sel = frame_ids[0]
        st.session_state.selected_frame_id = sel

    chart_theme = chart_theme_from_streamlit()

    # 帧选择放在双栏之上，避免左栏先于右栏执行时条带与热力图帧号不一致
    st.markdown("##### 当前帧")
    row_pick = st.columns([1, 2])
    with row_pick[0]:
        pick = st.selectbox(
            "帧 ID",
            options=frame_ids,
            index=frame_ids.index(int(sel)),
            format_func=lambda x: f"{int(x):05d}",
            help="也可在下方指标曲线中点击数据点切换；若未立即同步请稍候自动刷新。",
        )
    with row_pick[1]:
        st.metric("已选帧", f"{int(pick):05d}")

    if pick != st.session_state.selected_frame_id:
        st.session_state.selected_frame_id = int(pick)

    # 左栏更宽：约 62% / 38%
    col_chart, col_detail = st.columns([5, 3], gap="large")

    with col_chart:
        st.subheader("指标曲线")
        st.caption(
            "点击数据点可选中帧，与上方「帧 ID」及右侧热力图同步。"
            "切换应用深/浅色后若图表未变，请刷新页面。"
        )
        fig = metrics_figure(df, chart_theme, chart_height=820)
        event = st.plotly_chart(
            fig,
            width="stretch",
            theme=None,
            key="metric_chart",
            on_select="rerun",
            selection_mode="points",
            config={"scrollZoom": True, "displaylogo": False},
        )
        _apply_plotly_selection(event, frame_ids)

    sel = st.session_state.selected_frame_id
    if sel not in frame_ids:
        sel = frame_ids[0]
        st.session_state.selected_frame_id = sel

    fid = int(st.session_state.selected_frame_id)
    bundle = load_bundle_for_frame(fid, id_to_path, n_panels)
    if bundle is None:
        st.warning("无法加载该帧图像。")
        return

    inp, ref, recon, gt = (
        bundle.input_rgb,
        bundle.reference_rgb,
        bundle.reconstruction_rgb,
        bundle.gt_rgb,
    )

    with col_chart:
        st.markdown("##### 四宫格条带")
        st.caption("Input · Reference · Reconstruction · GT")
        st.plotly_chart(
            panel_strip_figure(inp, ref, recon, gt, chart_theme),
            width="stretch",
            theme=None,
            config={"displaylogo": False},
        )

    with col_detail:
        st.subheader("本帧详情")
        st.caption("与左侧条带为同一帧；在曲线中点击数据点可切换帧。")

        st.markdown("##### 绝对误差热力图")
        st.caption("Input–Recon | Recon–GT（通道均值）")
        h_ir = abs_diff_heatmap(inp, recon)
        h_rg = abs_diff_heatmap(recon, gt)
        st.plotly_chart(
            heatmap_figure(
                h_ir,
                h_rg,
                "|Input−Recon|（通道均值）",
                "|Recon−GT|（通道均值）",
                chart_theme,
            ),
            width="stretch",
            theme=None,
            config={"displaylogo": False},
        )

        st.markdown("##### SSIM 图")
        st.caption("Input–Recon | Recon–GT")
        smap_ir, _ = ssim_maps(inp, recon)
        smap_rg, _ = ssim_maps(recon, gt)
        st.plotly_chart(
            ssim_maps_figure(
                smap_ir,
                smap_rg,
                "SSIM map：Input–Recon",
                "SSIM map：Recon–GT",
                chart_theme,
            ),
            width="stretch",
            theme=None,
            config={"displaylogo": False},
        )

        row = df.loc[df["frame_id"] == float(fid)].iloc[0]
        with st.expander("本帧标量指标（与曲线一致）"):
            cols = st.columns(3)
            blocks = [
                ("Input–Recon", "ir"),
                ("Input–GT", "ig"),
                ("Recon–GT", "rg"),
            ]
            for i, (title, prefix) in enumerate(blocks):
                with cols[i]:
                    st.markdown(f"**{title}**")
                    st.write(f"MSE: {row[f'{prefix}_mse']:.6f}")
                    st.write(f"SSIM: {row[f'{prefix}_ssim']:.6f}")
                    st.write(f"PSNR: {row[f'{prefix}_psnr']:.3f} dB")
                    st.write(f"LPIPS: {row[f'{prefix}_lpips']:.6f}")


if __name__ == "__main__":
    main()
