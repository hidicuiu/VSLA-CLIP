"""空地智寻：基于 VSLA-CLIP 的跨平台视频行人检索演示界面。

启动方式（项目根目录）：
    conda run -n ReID python app.py

首次检索会加载模型并建立图库特征索引，后续查询会复用内存缓存。
界面提供 G2A 数据集样例和本地图片/视频上传两种查询入口。
"""

from __future__ import annotations

import argparse
import html
import os
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# 禁用 Gradio 遥测，避免离线环境或 Windows 证书异常影响本地启动。
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.io import loadmat


APP_DIR = Path(__file__).resolve().parent
DATA_ROOT = APP_DIR.parent / "G2A-VReID"
CONFIG_PATH = APP_DIR / "configs" / "adapter" / "vit_adapter_g2a_vifi_pbp_no_pbp_lr2e4.yml"
WEIGHT_PATH = APP_DIR / "output" / "G2A" / "vifi_pbp_ablate_no_pbp_lr2e4" / "ViT-B-16_mAP_best.pth"
MODEL_NAME = "VSLA-CLIP · ViT-B/16 · Seed 1234"

PLATFORM_NAMES = {0: "地面摄像机", 1: "无人机"}
PLATFORM_SHORT = {0: "GROUND", 1: "AERIAL"}
DIRECTION_G2A = "地面查询 → 无人机图库"
DIRECTION_A2G = "无人机查询 → 地面图库"
SEQ_LEN = 8
TOPK_MAX = 12


@dataclass(frozen=True)
class Tracklet:
    index: int
    paths: Tuple[str, ...]
    pid: int
    camid: int

    @property
    def label(self) -> str:
        return (
            f"Q-{self.index:04d}  ·  ID {self.pid:04d}  ·  "
            f"{PLATFORM_NAMES.get(self.camid, '未知平台')}  ·  {len(self.paths)} 帧"
        )


def _safe_int(value: object) -> int:
    return int(np.asarray(value).item())


@lru_cache(maxsize=1)
def load_tracklets() -> Tuple[List[Tracklet], List[Tracklet], int]:
    """Read the official G2A split without constructing a training DataLoader."""
    info_dir = DATA_ROOT / "info"
    required = [
        info_dir / "train_name.txt",
        info_dir / "test_name.txt",
        info_dir / "tracks_train_info.mat",
        info_dir / "tracks_test_info.mat",
        info_dir / "query_IDX.mat",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("G2A 数据不完整：" + "、".join(missing))

    test_names = [
        line.strip()
        for line in (info_dir / "test_name.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    track_test = loadmat(str(info_dir / "tracks_test_info.mat"))["track_test_info"]
    query_indices = loadmat(str(info_dir / "query_IDX.mat"))["query_IDX"].squeeze().astype(int) - 1
    query_index_set = set(query_indices.tolist())

    def build(meta_index: int, output_index: int) -> Tracklet:
        start, end, pid, camid = [_safe_int(v) for v in track_test[meta_index]]
        names = test_names[start - 1 : end]
        person_dir = names[0][:4]
        paths = tuple(str(DATA_ROOT / "bbox_test" / person_dir / name) for name in names)
        return Tracklet(output_index, paths, pid, camid - 1)

    queries = [build(meta_index, i) for i, meta_index in enumerate(query_indices.tolist())]
    gallery_meta = [i for i in range(len(track_test)) if i not in query_index_set]
    gallery = [build(meta_index, i) for i, meta_index in enumerate(gallery_meta)]

    train_meta = loadmat(str(info_dir / "tracks_train_info.mat"))["track_train_info"]
    num_train_ids = int(len(np.unique(train_meta[:, 2])))
    return queries, gallery, num_train_ids


def sparse_sample_indices(length: int, count: int = SEQ_LEN) -> List[int]:
    """按测试阶段策略将轨迹分块，并从每个时间块选取代表帧。"""
    if length <= 0:
        return []
    if length < count:
        return list(range(length)) + [length - 1] * (count - length)
    block_size = length // count
    return [i * block_size for i in range(count)]


def sparsely_sample(items: Sequence, count: int = SEQ_LEN) -> List:
    if not items:
        raise ValueError("输入中没有可用帧。")
    return [items[index] for index in sparse_sample_indices(len(items), count)]


def open_rgb(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def direction_camids(direction: str) -> Tuple[int, int]:
    """返回（查询平台，目标图库平台）。"""
    return (0, 1) if direction.startswith("地面") else (1, 0)


def platform_pool(camid: int) -> List[Tracklet]:
    ground_tracklets, aerial_tracklets, _ = load_tracklets()
    return ground_tracklets if camid == 0 else aerial_tracklets


def query_choices(direction: str = DIRECTION_G2A) -> List[str]:
    try:
        query_camid, _ = direction_camids(direction)
        return [tracklet.label for tracklet in platform_pool(query_camid)]
    except Exception:
        return ["G2A 数据集未就绪"]


def query_preview(direction: str, choice: str) -> Tuple[List[Tuple[Image.Image, str]], str]:
    try:
        query_camid, target_camid = direction_camids(direction)
        queries = platform_pool(query_camid)
        index = int(choice.split("Q-")[1].split()[0])
        tracklet = queries[index]
        sampled = sparsely_sample(tracklet.paths)
        gallery = [
            (open_rgb(path), f"关键帧 {i + 1:02d} / {SEQ_LEN:02d}")
            for i, path in enumerate(sampled)
        ]
        note = (
            f"<div class='source-note'><span class='source-dot'></span>"
            f"当前样例来自 <b>{PLATFORM_NAMES[tracklet.camid]}</b>，共 {len(tracklet.paths)} 帧；"
            f"系统将到 <b>{PLATFORM_NAMES[target_camid]}</b> 图库中检索同一目标。</div>"
        )
        return gallery, note
    except Exception as exc:
        return [], f"<div class='error-note'>预览失败：{html.escape(str(exc))}</div>"


def direction_changed(direction: str):
    choices = query_choices(direction)
    selected = choices[0]
    preview, note = query_preview(direction, selected)
    return gr.update(choices=choices, value=selected), preview, note


def _decode_video(path: str) -> List[Image.Image]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "当前环境缺少 opencv-python-headless，暂时无法解码视频；可上传多张连续图片。"
        ) from exc

    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError("无法打开该视频，请尝试 MP4、AVI 或 MOV 格式。")

    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    target_indices = set(sparse_sample_indices(total, SEQ_LEN))
    frames: List[Image.Image] = []
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if total <= 0 or frame_index in target_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
            if total > 0 and len(frames) >= SEQ_LEN:
                break
        frame_index += 1
    capture.release()
    if not frames:
        raise ValueError("视频中未读取到有效画面。")
    return sparsely_sample(frames)


def load_uploaded_frames(files: Optional[Sequence[str]]) -> List[Image.Image]:
    if not files:
        raise ValueError("请上传一段目标视频，或上传多张连续行人图片。")
    if isinstance(files, (str, Path)):
        files = [str(files)]

    image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    video_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg"}
    paths = [str(getattr(item, "name", item)) for item in files]
    video_paths = [path for path in paths if Path(path).suffix.lower() in video_suffixes]
    if video_paths:
        return _decode_video(video_paths[0])

    images = [open_rgb(path) for path in paths if Path(path).suffix.lower() in image_suffixes]
    if not images:
        raise ValueError("没有识别到支持的图片或视频文件。")
    return sparsely_sample(images)


def _text_font(size: int = 18) -> ImageFont.ImageFont:
    candidates = [
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "msyh.ttc",
        Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts" / "simhei.ttf",
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def contact_sheet(paths: Sequence[str], title: str = "") -> Image.Image:
    frames = [open_rgb(path) for path in sparsely_sample(paths, 4)]
    cell_w, cell_h, gap, header = 128, 256, 8, 42
    canvas = Image.new("RGB", (cell_w * 4 + gap * 3, cell_h + header), "#0b1628")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), title, fill="#dce9f8", font=_text_font(17))
    for i, frame in enumerate(frames):
        fitted = frame.resize((cell_w, cell_h), Image.Resampling.BICUBIC)
        canvas.paste(fitted, (i * (cell_w + gap), header))
    return canvas


class RetrievalRuntime:
    def __init__(self) -> None:
        self.model = None
        self.cfg = None
        self.device = None
        self._lock = threading.RLock()
        self._gallery_cache: Dict[Tuple[str, int], Tuple[object, List[Tracklet]]] = {}

    def load(self) -> None:
        if self.model is not None:
            return
        with self._lock:
            if self.model is not None:
                return
            if not CONFIG_PATH.exists() or not WEIGHT_PATH.exists():
                raise FileNotFoundError("模型配置或训练权重不存在。")

            import torch
            from config import cfg as base_cfg
            from model.make_model_clipvideoreid_reidadapter_pbp import make_model

            if not torch.cuda.is_available():
                raise RuntimeError("当前模型实现依赖 CUDA，但未检测到可用的 NVIDIA GPU。")

            _, _, num_train_ids = load_tracklets()
            local_cfg = base_cfg.clone()
            local_cfg.merge_from_file(str(CONFIG_PATH))
            local_cfg.SOLVER.SEED = 1234
            local_cfg.freeze()
            torch.cuda.set_device(0)
            model = make_model(local_cfg, num_class=num_train_ids, camera_num=2, view_num=0)
            model.load_param(str(WEIGHT_PATH))
            model.eval().cuda()

            self.cfg = local_cfg
            self.device = torch.device("cuda:0")
            self.model = model

    def _tensor_from_frames(self, frames: Sequence[Image.Image]):
        import torch

        if self.cfg is None:
            raise RuntimeError("模型尚未加载。")
        height, width = self.cfg.INPUT.SIZE_TEST
        mean = torch.tensor(self.cfg.INPUT.PIXEL_MEAN).view(3, 1, 1)
        std = torch.tensor(self.cfg.INPUT.PIXEL_STD).view(3, 1, 1)
        tensors = []
        for frame in sparsely_sample(frames, SEQ_LEN):
            resized = frame.convert("RGB").resize((width, height), Image.Resampling.BICUBIC)
            array = np.asarray(resized, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).permute(2, 0, 1)
            tensors.append((tensor - mean) / std)
        return torch.stack(tensors, dim=1)

    def _features(self, frame_groups: Sequence[Sequence[Image.Image]], camids: Sequence[int]):
        import torch
        import torch.nn.functional as functional

        self.load()
        batch = torch.stack([self._tensor_from_frames(frames) for frames in frame_groups], dim=0)
        camera = torch.tensor(camids, dtype=torch.long)
        with torch.inference_mode():
            feature = self.model(batch.to(self.device), cam_label=camera.to(self.device))
            feature = functional.normalize(feature.float(), p=2, dim=1)
        return feature.cpu()

    @staticmethod
    def _candidate_subset(tracklets: List[Tracklet], size: int) -> List[Tracklet]:
        if size <= 0 or size >= len(tracklets):
            return tracklets
        indices = np.linspace(0, len(tracklets) - 1, size).round().astype(int)
        return [tracklets[i] for i in indices]

    def gallery_features(
        self, query_camid: int, index_size: str, progress: gr.Progress
    ):
        _, target_camid = direction_camids(
            DIRECTION_G2A if query_camid == 0 else DIRECTION_A2G
        )
        gallery = platform_pool(target_camid)

        size_map = {"快速 · 240 段": 240, "均衡 · 600 段": 600, "完整图库": 0}
        size = size_map.get(index_size, 240)
        candidates = self._candidate_subset(gallery, size)
        cache_key = (f"gallery-cam{target_camid}", len(candidates))
        if cache_key in self._gallery_cache:
            return self._gallery_cache[cache_key]

        batches = []
        batch_size = 12
        progress(0.16, desc=f"首次构建图库索引（{len(candidates)} 段）")
        for start in range(0, len(candidates), batch_size):
            batch_items = candidates[start : start + batch_size]
            frame_groups = [[open_rgb(path) for path in sparsely_sample(item.paths)] for item in batch_items]
            batches.append(self._features(frame_groups, [item.camid for item in batch_items]))
            completed = min(len(candidates), start + len(batch_items))
            progress(0.16 + 0.62 * completed / len(candidates), desc=f"编码图库 {completed}/{len(candidates)}")

        import torch

        features = torch.cat(batches, dim=0)
        self._gallery_cache[cache_key] = (features, candidates)
        return features, candidates

    def search(
        self,
        query_frames: Sequence[Image.Image],
        query_camid: int,
        index_size: str,
        top_k: int,
        progress: gr.Progress,
    ) -> List[Tuple[Tracklet, float]]:
        import torch

        progress(0.05, desc="加载 VSLA-CLIP 大模型")
        with self._lock:
            query_feature = self._features([query_frames], [query_camid])
            gallery_feature, candidates = self.gallery_features(query_camid, index_size, progress)
            progress(0.84, desc="计算跨平台语义相似度")
            similarity = torch.matmul(query_feature, gallery_feature.transpose(0, 1))[0]
            values, indices = torch.topk(similarity, k=min(int(top_k), len(candidates)))
        return [(candidates[int(i)], float(score)) for score, i in zip(values, indices)]


RUNTIME = RetrievalRuntime()


def model_health() -> Tuple[str, str]:
    model_ok = WEIGHT_PATH.exists()
    data_ok = DATA_ROOT.exists()
    try:
        import torch

        gpu_ok = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if gpu_ok else "未检测到 CUDA"
    except Exception:
        gpu_ok = False
        gpu_name = "PyTorch 环境异常"

    state = "就绪" if model_ok and data_ok and gpu_ok else "需检查"
    detail = f"{MODEL_NAME} · {gpu_name}"
    css_class = "ready" if state == "就绪" else "warning"
    return state, f"<span class='status-pill {css_class}'><i></i>{state}</span><span class='device'>{html.escape(detail)}</span>"


def stage_html(elapsed: float, count: int, best_score: float) -> str:
    return f"""
    <div class="run-summary">
      <div><small>检索状态</small><strong>分析完成</strong></div>
      <div><small>候选规模</small><strong>{count:,} 段视频</strong></div>
      <div><small>最高余弦分</small><strong>{best_score:.4f}</strong></div>
      <div><small>本次耗时</small><strong>{elapsed:.2f} 秒</strong></div>
    </div>
    <div class="stage-line">
      <span class="done">01 视频采样</span><b>→</b><span class="done">02 视觉编码</span><b>→</b>
      <span class="done">03 VSLA 适配</span><b>→</b><span class="done">04 视频级聚合</span><b>→</b>
      <span class="done">05 Top-K 检索</span>
    </div>
    """


def result_table_html(results: Sequence[Tuple[Tracklet, float]]) -> str:
    rows = []
    for rank, (tracklet, score) in enumerate(results, 1):
        display_score = max(0.0, min(100.0, score * 100.0))
        rows.append(
            "<tr>"
            f"<td><span class='rank'>#{rank:02d}</span></td>"
            f"<td>候选 {tracklet.index:04d}</td>"
            f"<td>{PLATFORM_NAMES.get(tracklet.camid, '未知')}</td>"
            f"<td>{len(tracklet.paths)} 帧</td>"
            f"<td><div class='scorebar'><i style='width:{display_score:.1f}%'></i></div>"
            f"<b>{score:.4f}</b></td>"
            "</tr>"
        )
    return (
        "<div class='result-table-wrap'><table class='result-table'>"
        "<thead><tr><th>排序</th><th>轨迹片段</th><th>来源平台</th><th>帧数</th><th>余弦相似度</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
        "<p class='score-help'>相似度用于候选排序，不等同于身份概率；最终结果需由工作人员复核。</p></div>"
    )


def run_search(
    source_mode: str,
    direction: str,
    example_choice: str,
    upload_files: Optional[Sequence[str]],
    index_size: str,
    top_k: int,
    progress: gr.Progress = gr.Progress(),
):
    start = time.perf_counter()
    try:
        query_camid, target_camid = direction_camids(direction)
        if source_mode.startswith("系统示例"):
            queries = platform_pool(query_camid)
            query_index = int(example_choice.split("Q-")[1].split()[0])
            selected = queries[query_index]
            query_frames = [open_rgb(path) for path in sparsely_sample(selected.paths)]
            query_camid = selected.camid
            query_caption = f"系统示例 Q-{selected.index:04d} · {PLATFORM_NAMES[selected.camid]}"
        else:
            query_frames = load_uploaded_frames(upload_files)
            query_caption = f"用户上传 · {PLATFORM_NAMES[query_camid]}"

        results = RUNTIME.search(query_frames, query_camid, index_size, int(top_k), progress)
        elapsed = time.perf_counter() - start
        result_gallery = [
            (
                contact_sheet(
                    item.paths,
                    f"RANK {rank:02d}  |  {PLATFORM_SHORT.get(item.camid, 'UNKNOWN')}",
                ),
                f"#{rank:02d} · 候选 {item.index:04d} · 相似度 {score:.4f}",
            )
            for rank, (item, score) in enumerate(results, 1)
        ]
        sampled_gallery = [(frame, f"{query_caption} · 帧 {i + 1}") for i, frame in enumerate(query_frames)]
        query_note = (
            f"<div class='source-note'><span class='source-dot'></span>"
            f"本次查询：<b>{html.escape(query_caption)}</b>；已提取 {len(query_frames)} 个关键帧并完成视频级融合。</div>"
        )
        candidate_count = len(platform_pool(target_camid))
        size_map = {"快速 · 240 段": 240, "均衡 · 600 段": 600, "完整图库": candidate_count}
        actual_count = min(candidate_count, size_map.get(index_size, 240))
        summary = stage_html(elapsed, actual_count, results[0][1] if results else 0.0)
        table = result_table_html(results)
        return sampled_gallery, result_gallery, summary, table, query_note
    except Exception as exc:
        message = html.escape(str(exc))
        error = (
            "<div class='error-panel'><b>检索未完成</b>"
            f"<span>{message}</span><small>请检查模型权重、G2A 数据集、CUDA 环境或上传文件格式。</small></div>"
        )
        return [], [], error, "", f"<div class='error-note'>{message}</div>"


def source_visibility(mode: str):
    example_visible = mode.startswith("系统示例")
    return gr.update(visible=example_visible), gr.update(visible=not example_visible)


CSS = """
:root {
  --ink: #eaf3ff; --muted: #8fa6be; --panel: #0c1829; --panel2: #101f33;
  --line: #20344d; --cyan: #3ce7d1; --blue: #5f8cff; --orange: #ffb45c;
}
body, .gradio-container { background: #07111f !important; color: var(--ink) !important; }
.gradio-container { max-width: 1480px !important; font-family: "Microsoft YaHei UI", "PingFang SC", sans-serif !important; }
.app-shell { border: 1px solid #1c3048; background: #091523; border-radius: 18px; overflow: hidden; box-shadow: 0 20px 80px #0007; }
.hero { padding: 30px 34px 26px; background: linear-gradient(110deg, #0d2238, #0a1727 66%, #102d39); border-bottom: 1px solid var(--line); }
.hero-top { display: flex; align-items: center; justify-content: space-between; gap: 20px; }
.brand { display:flex; align-items:center; gap:16px; }
.brand-mark { width:52px; height:52px; border:1px solid #3ce7d177; border-radius:14px; display:grid; place-items:center; background:#3ce7d112; }
.brand-mark svg { width:30px; height:30px; stroke:var(--cyan); fill:none; stroke-width:1.7; }
.eyebrow { margin:0 0 6px; color:var(--cyan); letter-spacing:.18em; font-size:12px; font-weight:700; }
.hero h1 { margin:0; font-size:29px; line-height:1.2; color:#f5f9ff; letter-spacing:.04em; }
.hero h1 em { font-style:normal; color:#94a9bf; font-weight:400; font-size:16px; margin-left:10px; }
.hero-sub { margin:18px 0 0; max-width:930px; color:#a9bbcf; font-size:14px; line-height:1.8; }
.status-wrap { min-width:275px; display:flex; align-items:center; justify-content:flex-end; gap:10px; }
.status-pill { display:inline-flex; align-items:center; gap:7px; padding:8px 12px; border:1px solid #3ce7d155; border-radius:999px; color:var(--cyan); background:#3ce7d112; font-size:12px; font-weight:700; }
.status-pill i { width:7px; height:7px; border-radius:50%; background:var(--cyan); box-shadow:0 0 10px var(--cyan); }
.status-pill.warning { color:var(--orange); border-color:#ffb45c55; background:#ffb45c12; }
.status-pill.warning i { background:var(--orange); }
.device { display:block; color:#7890a8; font-size:11px; margin-top:6px; text-align:right; }
.metric-strip { display:grid; grid-template-columns:repeat(4, 1fr); gap:1px; background:var(--line); border-bottom:1px solid var(--line); }
.metric-strip div { background:#0a1727; padding:16px 22px; }
.metric-strip small { display:block; color:#7891aa; font-size:11px; letter-spacing:.06em; }
.metric-strip b { display:block; color:#e9f2fd; font-size:18px; margin-top:4px; }
.metric-strip b.accent { color:var(--cyan); }
.main-tabs > .tab-nav { background:#0a1727 !important; border-bottom:1px solid var(--line) !important; padding:0 18px !important; }
.main-tabs > .tab-nav button { color:#8fa6be !important; border:none !important; padding:15px 20px !important; }
.main-tabs > .tab-nav button.selected { color:var(--cyan) !important; border-bottom:2px solid var(--cyan) !important; }
.control-card, .output-card { border:1px solid var(--line) !important; border-radius:14px !important; background:var(--panel) !important; padding:16px !important; }
.output-card > .styler, .control-card > .styler { background:var(--panel) !important; }
.section-kicker { color:var(--cyan); font-size:11px; letter-spacing:.15em; font-weight:700; margin-bottom:4px; }
.section-title { font-size:18px; font-weight:700; color:#edf5ff; margin-bottom:4px; }
.section-desc { color:#8299b1; font-size:12px; line-height:1.6; margin-bottom:14px; }
.source-note { border-left:2px solid var(--cyan); padding:10px 12px; background:#3ce7d10c; color:#9db1c5; font-size:12px; border-radius:0 8px 8px 0; }
.source-dot { display:inline-block; width:6px; height:6px; border-radius:50%; background:var(--cyan); margin-right:8px; }
.primary-action { background:var(--cyan) !important; color:#06151a !important; border:0 !important; font-weight:800 !important; min-height:46px !important; }
.primary-action:hover { background:#6df4e3 !important; }
.run-summary { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:6px 0 14px; }
.run-summary div { padding:12px 14px; background:#101f33; border:1px solid var(--line); border-radius:10px; }
.run-summary small { display:block; color:#7890aa; font-size:10px; margin-bottom:4px; }
.run-summary strong { color:#eaf3ff; font-size:14px; }
.stage-line { display:flex; flex-wrap:wrap; align-items:center; gap:7px; padding:12px 14px; border:1px solid #274057; background:#07131f; border-radius:10px; font-size:11px; color:#668099; }
.stage-line .done { color:var(--cyan); }
.stage-line b { color:#45647e; }
.result-table-wrap { overflow-x:auto; border:1px solid var(--line); border-radius:12px; }
.result-table { width:100%; border-collapse:collapse; font-size:12px; }
.result-table th { background:#102036; color:#8199b0; text-align:left; padding:11px 12px; font-weight:600; }
.result-table td { border-top:1px solid #1c3048; padding:10px 12px; color:#bdcad8; }
.rank { color:var(--cyan); font-weight:800; }
.scorebar { width:90px; height:4px; display:inline-block; margin-right:9px; background:#1b2e43; border-radius:6px; vertical-align:middle; overflow:hidden; }
.scorebar i { display:block; height:100%; background:linear-gradient(90deg, #5f8cff, #3ce7d1); }
.result-table b { color:#dfeaff; font-variant-numeric:tabular-nums; }
.score-help { padding:10px 12px 0; color:#71879d; font-size:10px; }
.error-panel, .error-note { padding:14px; border:1px solid #ff8a7055; background:#ff8a7010; border-radius:10px; color:#ffb19f; }
.error-panel b, .error-panel span, .error-panel small { display:block; margin-bottom:5px; }
.tech-grid { display:grid; grid-template-columns:repeat(5,1fr); gap:10px; margin:14px 0; }
.tech-node { position:relative; padding:16px 14px; min-height:104px; background:#0c1b2d; border:1px solid var(--line); border-radius:12px; }
.tech-node:not(:last-child):after { content:'›'; position:absolute; right:-9px; top:40%; color:var(--cyan); font-size:22px; z-index:2; }
.tech-node i { color:var(--cyan); font-style:normal; font-size:11px; letter-spacing:.08em; }
.tech-node b { display:block; margin:8px 0 5px; color:#edf5ff; font-size:14px; }
.tech-node span { color:#7f96ad; font-size:11px; line-height:1.5; }
.benchmark { display:grid; grid-template-columns:1.2fr 1fr; gap:16px; margin-top:16px; }
.benchmark-card { border:1px solid var(--line); background:#0c1a2c; border-radius:14px; padding:20px; }
.benchmark-card h3 { margin:0 0 16px; font-size:15px; color:#e8f2ff; }
.bar-row { display:grid; grid-template-columns:70px 1fr 56px; gap:10px; align-items:center; margin:12px 0; font-size:12px; color:#8fa6bd; }
.bar { height:8px; background:#1b3047; border-radius:9px; overflow:hidden; }
.bar i { display:block; height:100%; background:linear-gradient(90deg,#537dff,#3ce7d1); }
.bar-row strong { color:#e9f3ff; text-align:right; }
.feature-list { display:grid; gap:10px; }
.feature-list div { padding:12px; border-left:2px solid var(--blue); background:#0a1626; color:#8da3ba; font-size:12px; line-height:1.6; }
.feature-list b { display:block; color:#dbe8f7; margin-bottom:2px; }
.ethics { padding:18px; border:1px solid #ffb45c44; background:#ffb45c0a; border-radius:12px; color:#aebdca; line-height:1.8; font-size:12px; }
.ethics b { color:var(--orange); }
.footer-note { text-align:center; color:#526c85; font-size:10px; padding:18px 0 4px; letter-spacing:.08em; }
label, .block-label { color:#adbed0 !important; }
.gradio-container input, .gradio-container textarea { color:#eaf3ff !important; }
.gallery-item { background:#07111f !important; border-color:#1f354d !important; }
.result-gallery {
  height:760px !important;
  min-height:760px !important;
  max-height:760px !important;
  overflow:hidden !important;
}
.result-gallery .gallery-container {
  height:100% !important;
  min-height:0 !important;
  overflow:hidden !important;
}
.result-gallery .grid-wrap {
  height:100% !important;
  min-height:0 !important;
  max-height:none !important;
  overflow-x:hidden !important;
  overflow-y:auto !important;
  overscroll-behavior:contain;
  scrollbar-gutter:stable;
  scrollbar-color:#3ce7d1 #102036;
  scrollbar-width:thin;
}
.result-gallery .grid-wrap::-webkit-scrollbar { width:8px; }
.result-gallery .grid-wrap::-webkit-scrollbar-track { background:#102036; }
.result-gallery .grid-wrap::-webkit-scrollbar-thumb { background:#3ce7d1; border-radius:8px; }
.result-gallery .grid-container {
  height:auto !important;
  min-height:0 !important;
  max-height:none !important;
  overflow:visible !important;
}
.result-gallery .thumbnail-item {
  aspect-ratio:536 / 298 !important;
  height:auto !important;
  min-height:0 !important;
}
.result-gallery .thumbnail-item img {
  width:100% !important;
  height:auto !important;
  object-fit:contain !important;
}
@media (max-width: 900px) {
  .hero-top { align-items:flex-start; flex-direction:column; }
  .status-wrap { justify-content:flex-start; }
  .device { text-align:left; }
  .metric-strip, .run-summary { grid-template-columns:repeat(2,1fr); }
  .tech-grid { grid-template-columns:1fr; }
  .tech-node:after { display:none; }
  .benchmark { grid-template-columns:1fr; }
}
"""


THEME = gr.themes.Base(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.blue,
    neutral_hue=gr.themes.colors.slate,
).set(
    body_background_fill="#07111f",
    body_background_fill_dark="#07111f",
    body_text_color="#eaf3ff",
    body_text_color_dark="#eaf3ff",
    body_text_color_subdued="#8fa6be",
    body_text_color_subdued_dark="#8fa6be",
    background_fill_primary="#0c1829",
    background_fill_primary_dark="#0c1829",
    background_fill_secondary="#101f33",
    background_fill_secondary_dark="#101f33",
    block_background_fill="#0c1829",
    block_background_fill_dark="#0c1829",
    block_border_color="#20344d",
    block_border_color_dark="#20344d",
    block_label_background_fill="#101f33",
    block_label_background_fill_dark="#101f33",
    block_label_text_color="#adbed0",
    block_label_text_color_dark="#adbed0",
    input_background_fill="#101f33",
    input_background_fill_dark="#101f33",
    input_background_fill_focus="#13263d",
    input_background_fill_focus_dark="#13263d",
    input_border_color="#2a425b",
    input_border_color_dark="#2a425b",
    input_border_color_focus="#3ce7d1",
    input_border_color_focus_dark="#3ce7d1",
    checkbox_label_background_fill="#101f33",
    checkbox_label_background_fill_dark="#101f33",
    checkbox_label_background_fill_selected="#153b41",
    checkbox_label_background_fill_selected_dark="#153b41",
    checkbox_label_border_color="#2a425b",
    checkbox_label_border_color_dark="#2a425b",
    checkbox_label_text_color="#aebed0",
    checkbox_label_text_color_dark="#aebed0",
    checkbox_label_text_color_selected="#76f4e4",
    checkbox_label_text_color_selected_dark="#76f4e4",
    panel_background_fill="#0a1727",
    panel_background_fill_dark="#0a1727",
    button_primary_background_fill="#3ce7d1",
    button_primary_background_fill_dark="#3ce7d1",
    button_primary_background_fill_hover="#6df4e3",
    button_primary_background_fill_hover_dark="#6df4e3",
    button_primary_text_color="#06151a",
    button_primary_text_color_dark="#06151a",
)


def header_html() -> str:
    _, health = model_health()
    return f"""
    <div class="app-shell">
      <header class="hero">
        <div class="hero-top">
          <div class="brand">
            <div class="brand-mark">
              <svg viewBox="0 0 32 32" aria-hidden="true"><path d="M4 22l12-15 12 15M8 18h16M11 22l5 5 5-5"/><circle cx="16" cy="12" r="3"/></svg>
            </div>
            <div><p class="eyebrow">AIR–GROUND INTELLIGENCE</p><h1>空地智寻 <em>跨平台视频行人智能检索系统</em></h1></div>
          </div>
          <div class="status-wrap"><div>{health}</div></div>
        </div>
        <p class="hero-sub">以 CLIP 视觉语言大模型为语义底座，融合视频多帧身份线索与集合级适配，在地面摄像机和无人机视频之间完成目标候选检索。</p>
      </header>
      <div class="metric-strip">
        <div><small>MODEL</small><b>ViT-B/16 + VSLA</b></div>
        <div><small>BENCHMARK · G2A</small><b class="accent">mAP 81.75%</b></div>
        <div><small>RANK-1</small><b>75.03%</b></div>
        <div><small>INPUT</small><b>8-frame Video</b></div>
      </div>
    </div>
    """


TECH_HTML = """
<div class="control-card">
  <div class="section-kicker">MODEL PIPELINE</div><div class="section-title">VSLA-CLIP 多模态训练与检索链路</div>
  <div class="section-desc">训练阶段在 CLIP 图文空间中完成身份视觉与语义对齐，检索阶段通过视觉编码和视频集合级适配生成可比对的视频特征。</div>
  <div class="tech-grid">
    <div class="tech-node"><i>01 / INPUT</i><b>视频帧采样</b><span>将目标轨迹划分为 8 个时间区段，稀疏选取代表帧以覆盖姿态与视角变化。</span></div>
    <div class="tech-node"><i>02 / CLIP</i><b>CLIP 视觉编码</b><span>由 ViFi-CLIP 初始化的 ViT-B/16 提取人体颜色、纹理和结构等帧级视觉嵌入。</span></div>
    <div class="tech-node"><i>03 / ALIGNMENT</i><b>身份视觉—语义对齐</b><span>训练阶段学习身份描述与共享文本提示，使视频视觉表征获得稳定的身份语义约束。</span></div>
    <div class="tech-node"><i>04 / VSLA</i><b>视频集合级适配</b><span>IFA 适配帧内外观表征，CFAA 交换跨帧互补信息，经均值池化形成视频级特征。</span></div>
    <div class="tech-node"><i>05 / RETRIEVAL</i><b>测试 / 推理：余弦检索</b><span>模型训练完成后，对归一化的查询与图库视频特征进行相似度排序，输出可复核的 Top-K 候选。</span></div>
  </div>
  <div class="benchmark">
    <div class="benchmark-card"><h3>G2A 测试集表现 · Seed 1234</h3>
      <div class="bar-row"><span>mAP</span><div class="bar"><i style="width:81.75%"></i></div><strong>81.75%</strong></div>
      <div class="bar-row"><span>Rank-1</span><div class="bar"><i style="width:75.03%"></i></div><strong>75.03%</strong></div>
      <div class="bar-row"><span>Rank-5</span><div class="bar"><i style="width:90.04%"></i></div><strong>90.04%</strong></div>
      <div class="bar-row"><span>Rank-10</span><div class="bar"><i style="width:93.33%"></i></div><strong>93.33%</strong></div>
    </div>
    <div class="benchmark-card"><h3>作品创新点</h3><div class="feature-list">
      <div><b>视觉 + 语言</b>训练阶段以身份语义约束视觉表征，增强跨场景身份辨识能力。</div>
      <div><b>帧内 + 跨帧</b>利用视频集合中的互补信息抵抗遮挡、模糊和姿态变化。</div>
      <div><b>地面 + 低空</b>依托视觉—语义对齐与视频集合级建模缓解俯视角、尺度和背景域差异。</div>
    </div></div>
  </div>
</div>
"""


ABOUT_HTML = """
<div class="control-card">
  <div class="section-kicker">APPLICATION</div><div class="section-title">面向真实业务的辅助检索原型</div>
  <div class="benchmark">
    <div class="benchmark-card"><h3>典型使用场景</h3><div class="feature-list">
      <div><b>智慧园区</b>目标离开地面摄像机覆盖区后，通过无人机视频继续寻找候选轨迹。</div>
      <div><b>应急搜寻</b>将现场采集的人员片段与空地视频库快速比对，缩小人工排查范围。</div>
      <div><b>视频内容检索</b>按目标外观语义聚合跨摄像机、跨平台的相关视频片段。</div>
    </div></div>
    <div class="benchmark-card"><h3>系统边界</h3>
      <div class="ethics"><b>本系统是候选排序工具，不作身份认定。</b><br>检索结果应由授权人员结合时间、地点和原始视频复核；部署时应遵守数据最小化、访问控制、操作留痕和定期删除要求，不得用于无授权监控或自动执法。</div>
    </div>
  </div>
  <div class="footer-note">空地智寻 · AI 大模型创新应用竞赛作品原型 · THEME 03 MULTIMODAL FUSION</div>
</div>
"""


def build_demo() -> gr.Blocks:
    initial_direction = DIRECTION_G2A
    choices = query_choices(initial_direction)
    initial_choice = choices[0]
    initial_preview, initial_note = query_preview(initial_direction, initial_choice)

    with gr.Blocks(
        css=CSS,
        theme=THEME,
        title="空地智寻 · 跨平台视频行人智能检索",
        analytics_enabled=False,
    ) as demo:
        gr.HTML(header_html())
        with gr.Tabs(elem_classes=["main-tabs"]):
            with gr.Tab("智能检索"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=4, min_width=350, elem_classes=["control-card"]):
                        gr.HTML("<div class='section-kicker'>QUERY SETUP</div><div class='section-title'>设置检索任务</div><div class='section-desc'>选择官方样例，或上传一段目标视频 / 多张连续图片。</div>")
                        source_mode = gr.Radio(
                            ["系统示例 · G2A", "上传本地视频 / 图片"],
                            value="系统示例 · G2A",
                            label="查询来源",
                        )
                        direction = gr.Radio(
                            [DIRECTION_G2A, DIRECTION_A2G],
                            value=initial_direction,
                            label="检索方向",
                        )
                        with gr.Group(visible=True) as example_group:
                            example_choice = gr.Dropdown(choices, value=initial_choice, label="目标轨迹")
                        with gr.Group(visible=False) as upload_group:
                            upload_files = gr.File(
                                label="目标视频或连续图片",
                                file_count="multiple",
                                file_types=["image", "video"],
                                type="filepath",
                            )
                        index_size = gr.Radio(
                            ["快速 · 240 段", "均衡 · 600 段", "完整图库"],
                            value="快速 · 240 段",
                            label="图库规模",
                        )
                        top_k = gr.Slider(3, TOPK_MAX, value=6, step=1, label="返回候选数 Top-K")
                        run_button = gr.Button("开始跨平台检索  →", variant="primary", elem_classes=["primary-action"])
                        gr.HTML("<div class='section-desc' style='margin-top:10px'>首次运行需加载模型并建立目标平台索引；同一检索方向的后续查询将直接复用缓存。</div>")

                    with gr.Column(scale=8, min_width=600):
                        with gr.Group(elem_classes=["output-card"]):
                            gr.HTML("<div class='section-kicker'>QUERY EVIDENCE</div><div class='section-title'>多帧查询证据</div>")
                            query_gallery = gr.Gallery(
                                value=initial_preview,
                                columns=4,
                                rows=2,
                                height=410,
                                object_fit="contain",
                                show_label=False,
                            )
                            source_note = gr.HTML(initial_note)

                with gr.Group(elem_classes=["output-card"]):
                    gr.HTML("<div class='section-kicker'>RETRIEVAL RESULTS</div><div class='section-title'>跨平台候选结果</div><div class='section-desc'>结果按视频级语义余弦相似度排序，每张候选卡展示 4 个跨帧证据。</div>")
                    run_summary = gr.HTML("<div class='stage-line'>等待发起检索任务</div>")
                    result_gallery = gr.Gallery(
                        columns=2,
                        rows=3,
                        height=760,
                        object_fit="contain",
                        show_label=False,
                        elem_classes=["result-gallery"],
                    )
                    result_table = gr.HTML("")

                source_mode.change(
                    source_visibility,
                    inputs=source_mode,
                    outputs=[example_group, upload_group],
                    show_progress="hidden",
                )
                direction.change(
                    direction_changed,
                    inputs=direction,
                    outputs=[example_choice, query_gallery, source_note],
                    show_progress="hidden",
                )
                example_choice.change(
                    query_preview,
                    inputs=[direction, example_choice],
                    outputs=[query_gallery, source_note],
                    show_progress="hidden",
                )
                run_button.click(
                    run_search,
                    inputs=[source_mode, direction, example_choice, upload_files, index_size, top_k],
                    outputs=[query_gallery, result_gallery, run_summary, result_table, source_note],
                )

            with gr.Tab("模型看板"):
                gr.HTML(TECH_HTML)
            with gr.Tab("应用说明"):
                gr.HTML(ABOUT_HTML)

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="空地智寻 Gradio 演示系统")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument(
        "--server-port",
        type=int,
        default=None,
        help="指定服务端口；不指定时从 7860 开始自动选择空闲端口",
    )
    parser.add_argument("--share", action="store_true", help="创建临时公网分享链接")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_demo().queue(default_concurrency_limit=1).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        show_error=True,
    )
