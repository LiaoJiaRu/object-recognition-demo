"""app.py

Gradio 图形界面（阶段 8）。
- 输入目录路径 -> 自动扫描子文件夹批量录入特征库
- 上传图片 + 名称 -> 单个录入特征库
- 上传图片 -> 整图识别 -> 名称 + 相似度展示
- 上传视频 -> 抽帧 -> 逐帧整帧识别 -> 帧画廊 + 多数投票聚合
- 查看已注册类别列表
- 按名称删除类别

后端只调用 core 中已有模块（FeatureExtractor / FeatureGallery / Recognizer /
core.video），本文件不重复实现任何特征或识别逻辑。

视频识别与图片识别共用同一特征库，
类别管理（录入/删除）请在「录入」「管理」Tab 操作。

启动:
    python app.py
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import yaml
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from core.feature_extractor import FeatureExtractor  # noqa: E402
from core.gallery import FeatureGallery  # noqa: E402
from core.recognizer import Recognizer  # noqa: E402
from core.video import (  # noqa: E402
    DEFAULT_FRAME_INTERVAL,
    aggregate_results,
    sample_frames,
)
from core.webcam import (  # noqa: E402
    COLOR_GREEN,
    COLOR_RED,
    crop_guide_box_roi,
    draw_subtitle_bar,
    get_guide_box_rect,
    recognize_frame,
)

DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


class RecognitionService:
    """界面与 core 模块之间的薄封装：持有模型/特征库单例，供各回调复用。

    图片识别与视频识别共用同一个 extractor / gallery / recognizer 实例，
    避免重复加载模型。
    """

    # 支持的图片扩展名（小写匹配）
    _IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".jfif", ".webp", ".bmp"}

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.extractor = FeatureExtractor(
            device=cfg.get("device", "cpu"),
            image_size=cfg.get("image_size", 224),
        )
        self.gallery = FeatureGallery(cfg, extractor=self.extractor)
        if self.gallery.feature_path.exists() and self.gallery.labels_path.exists():
            self.gallery.load()
        self.recognizer = Recognizer(self.gallery, extractor=self.extractor, cfg=cfg)
        # 特征库读写互斥锁：Gradio 各事件（录入/删除/3 秒热加载）跑在不同线程，
        # 所有「读-改-写」gallery 的操作必须持锁，否则 load() 会覆盖未落盘的修改。
        self.gallery_lock = threading.Lock()
        # ---- 视频识别相关配置 ----
        self.unknown_label = cfg.get("unknown_label", "unknown")
        self.frame_interval = float(cfg.get("frame_interval", DEFAULT_FRAME_INTERVAL))
        self.max_frames = int(cfg.get("max_frames", 0))

    # ------------------------------------------------------------------
    # 图片识别：整图检索识别
    # ------------------------------------------------------------------
    def detect_and_recognize(
        self, image_path: str | None
    ) -> tuple[np.ndarray | None, str, str, object, str]:
        """对整张图片做特征检索识别。

        返回 (原图, 结果文本, 播报文本, unknown 提示区可见性更新, unknown 提示文案)。
        识别为 unknown 时提示区可见并给出提示文案，否则提示区隐藏、文案为空。
        """
        import time

        if not image_path or not Path(image_path).is_file():
            return None, "请先上传一张图片。", "", gr.update(visible=False), ""

        img_pil = Image.open(image_path).convert("RGB")
        result_md = self.recognize(image_path)
        label, score, _topk = self.recognizer.predict(str(image_path))
        if label == self.unknown_label:
            speak_text = self.cfg.get("unknown_label_zh", "未能识别出已知物体")
            prompt_md = (
                f"⚠️ **未能识别这张图片**：最高相似度 {score:.4f}，低于阈值 "
                f"{self.cfg.get('threshold', 0.75)}。\n\n"
                "如果这是你想让系统认识的物体，可以把它录入成新类别，之后就能识别出来。"
            )
            prompt_update = gr.update(visible=True)
        else:
            speak_text = f"识别结果，这可能是{label}，相似度{int(score * 100)}%"
            prompt_md = ""
            prompt_update = gr.update(visible=False)
        ts = str(int(time.time() * 1000))
        return np.array(img_pil), result_md, f"{ts}|{speak_text}", prompt_update, prompt_md

    # ------------------------------------------------------------------
    # 视频识别：逐帧整帧检索识别 + 多数投票画廊
    # ------------------------------------------------------------------
    def detect_and_recognize_video(
        self, video
    ) -> tuple[str, list[list], list]:
        """视频逐帧整帧识别 + 多数投票。返回 (结果 Markdown, 逐帧明细表, 帧画廊)。

        流程：
        1. 按固定间隔抽帧
        2. 每帧直接整帧提特征 + 余弦相似度检索
        3. 逐帧标签多数投票得到整段结果
        4. 帧以画廊形式展示（caption 标注该帧标签与相似度）
        """
        path = self._extract_video_path(video)
        if not path:
            return "请先上传一个视频文件。", [], []

        if len(self.gallery) == 0:
            return (
                "特征库为空，所有帧都会识别为 unknown。"
                "请先在「录入」Tab 录入类别。",
                [],
                [],
            )

        # ---- 1. 抽帧 ----
        try:
            sampled = sample_frames(
                path,
                interval_sec=self.frame_interval,
                max_frames=self.max_frames,
            )
        except (FileNotFoundError, ValueError) as exc:
            return f"视频读取失败：{exc}", [], []

        if not sampled:
            return "未能从视频中抽到任何帧（视频可能为空或损坏）。", [], []

        # ---- 2. 逐帧整帧识别 ----
        frame_results: list[tuple[str, float, list]] = []  # 供 aggregate_results 投票
        frames_detail: list[dict] = []  # 逐帧明细表
        annotated_frames: list[tuple[np.ndarray, str]] = []  # 画廊图片

        for ts, frame_bgr in sampled:
            label, score, topk = recognize_frame(
                self.recognizer, self.extractor, frame_bgr
            )
            frame_results.append((label, score, topk))
            frames_detail.append({
                "time": round(ts, 3),
                "label": label,
                "score": float(score),
            })

            # BGR -> RGB，作为画廊图片（不再画框）
            frame_rgb = frame_bgr[:, :, ::-1].copy()
            caption = f"t={ts:.2f}s: {label}({score:.2f})"
            annotated_frames.append((frame_rgb, caption))

        # ---- 3. 多数投票聚合 ----
        label, score, votes = aggregate_results(
            frame_results, unknown_label=self.unknown_label
        )

        votes_text = "，".join(
            f"{l} {n} 帧" for l, n in
            sorted(votes.items(), key=lambda kv: -kv[1])
        )
        lines = [
            "**识别结果**",
            "",
            f"- label: **{label}**",
            f"- score: **{score:.4f}**（获胜类别各帧均值）",
            f"- threshold: {self.cfg.get('threshold', 0.75)}",
            f"- 抽帧: 每 {self.frame_interval}s 1 帧，共 {len(sampled)} 帧",
            f"- 投票分布: {votes_text}",
        ]
        table = [
            [f"{f['time']:.2f}", f["label"], f"{f['score']:.4f}"]
            for f in frames_detail
        ]

        return "\n".join(lines), table, annotated_frames

    # ------------------------------------------------------------------
    # 录入
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_paths(files) -> list[str]:
        """把 gr.File 多文件上传的输出统一转成存在的文件路径列表。

        兼容 Gradio 返回 str 路径、文件对象（含 .name）或它们的列表。
        """
        if files is None:
            return []
        if isinstance(files, str):
            return [files] if Path(files).is_file() else []
        if isinstance(files, (list, tuple)):
            out: list[str] = []
            for f in files:
                if isinstance(f, str):
                    p = f
                else:
                    p = getattr(f, "name", None) or str(f)
                if p and Path(p).is_file():
                    out.append(p)
            return out
        return []

    @staticmethod
    def _check_name(name: str) -> str | None:
        """类别名（物品名）合法性校验；非法返回错误信息，合法返回 None。"""
        if not name:
            return "请先输入类别名称。"
        if name in (".", "..") or any(ch in name for ch in '\\/:*?"<>|'):
            return (
                f"类别名 '{name}' 含非法字符，请改用中文、字母、数字、空格或下划线。"
            )
        return None

    def enroll(self, images, name: str | None) -> tuple[str, str, list[str]]:
        """单个录入：上传 1~10 张图片（同一物体）+ 类别名。

        只把图片特征写入特征库，不复制也不改动任何图片文件。
        返回 (状态信息, 刷新后的列表文本, 本次上传的图片路径列表)；失败时路径列表为空。
        """
        name = (name or "").strip()
        err = self._check_name(name)
        if err:
            return err, self.list_text(), []

        paths = self._extract_paths(images)
        if not paths:
            return "请先上传至少一张图片。", self.list_text(), []
        if len(paths) > 10:
            return f"一次最多录入 10 张图片，当前上传了 {len(paths)} 张。", self.list_text(), []

        with self.gallery_lock:
            added = self.gallery.add(name, paths)
            self.gallery.save()
            counts = self.gallery.counts()
        status = (
            f"录入成功：类别 '{name}' 新增 {added} 条特征，"
            f"该类别累计 {counts.get(name, 0)} 条。"
        )
        return status, self.list_text(), list(paths)

    def enroll_paths_chunked(
        self, name: str | None, paths: list[str],
    ) -> tuple[bool, str]:
        """按单个录入逻辑把多张图片路径录入类别 name（供「学习 / 纠错」面板）。

        与单个录入（enroll）使用完全相同的名称校验与 gallery.add 入库代码，
        区别仅在于：不限制单次张数，内部自动按每批最多 10 张分批提特征，
        最后统一落盘一次。返回 (是否成功, 状态文本)。
        """
        name = (name or "").strip()
        err = self._check_name(name)
        if err:
            return False, err
        if not paths:
            return False, "没有可录入的图片。"
        with self.gallery_lock:
            total = 0
            for i in range(0, len(paths), 10):
                total += self.gallery.add(name, paths[i:i + 10])
            self.gallery.save()
            counts = self.gallery.counts()
        return True, (
            f"录入成功：类别 '{name}' 新增 {total} 条特征，"
            f"该类别累计 {counts.get(name, 0)} 条。"
        )

    def batch_enroll(
        self, dir_path: str | None, progress=gr.Progress(),
    ) -> Iterator[tuple[str, str]]:
        """批量录入（生成器）：扫描目录下的子文件夹，每个子文件夹名作为类别名，
        子文件夹内的图片自动录入该类别。

        目录结构示例：
            gallery/
              apple/   <- 类别名
                1.jpg   <- 该类别的图片
                2.jpg
              banana/
                1.jpg
                ...

        逐个类别录入并实时产出 (状态信息, 刷新后的列表文本)，
        配合 gr.Progress 在界面上显示进度条；Gradio 会在按钮与输出区
        自动叠加加载动画。返回值按 yield 流式送出。
        """
        if not dir_path or not dir_path.strip():
            yield "请先输入包含类别子文件夹的目录路径。", self.list_text()
            return

        root = Path(dir_path.strip())
        if not root.is_dir():
            yield f"目录不存在：{root}", self.list_text()
            return

        # 扫描所有子文件夹（仅一层，按名称排序）
        subdirs = sorted(
            [d for d in root.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )
        if not subdirs:
            yield f"目录下没有子文件夹：{root}", self.list_text()
            return

        # 预先收集每个子文件夹下的图片（跳过没有图片的子文件夹）
        plan: list[tuple[Path, list[Path]]] = []
        for subdir in subdirs:
            images = sorted(
                [f for f in subdir.iterdir()
                 if f.is_file() and f.suffix.lower() in self._IMAGE_EXTS],
            )
            if images:
                plan.append((subdir, images))

        if not plan:
            yield (
                f"目录 {root} 下的子文件夹中没有找到任何图片文件。\n\n"
                f"支持的格式：{', '.join(sorted(self._IMAGE_EXTS))}",
                self.list_text(),
            )
            return

        # 逐个类别录入，实时产出进度与状态。
        # 每个类别录完立即持锁落盘：后台 3 秒热加载定时器会调用 gallery.load()
        # 用磁盘文件覆盖内存库，若只在最后统一 save()，中途 load() 会清掉
        # 尚未落盘的类别（表现为"只注册了一个物体"）。逐类落盘后 load() 无害。
        details: list[str] = []
        total_added = 0
        for i, (subdir, images) in enumerate(plan):
            progress(
                i / len(plan),
                desc=f"正在录入 {subdir.name}（{i + 1}/{len(plan)} 个类别）",
            )
            with self.gallery_lock:
                added = self.gallery.add(subdir.name, images)
                self.gallery.save()
            total_added += added
            details.append(f"- {subdir.name}: {added} 张")
            yield (
                f"**正在批量录入**：已完成 {i + 1}/{len(plan)} 个类别，"
                f"新增 {total_added} 条特征。\n\n" + "\n".join(details),
                self.list_text(),
            )

        progress(1.0, desc="批量录入完成")
        yield (
            f"**批量录入完成**：共 {len(plan)} 个类别，"
            f"新增 {total_added} 条特征。\n\n" + "\n".join(details),
            self.list_text(),
        )

    # ------------------------------------------------------------------
    # 整图识别（图片 Tab 复用；亦供其它入口直接调用）
    # ------------------------------------------------------------------
    def recognize(self, image_path: str | None) -> str:
        if not image_path or not Path(image_path).is_file():
            return "请先上传一张图片。"
        label, score, topk = self.recognizer.predict(str(image_path))
        lines = [
            "**识别结果**",
            "",
            f"- label: **{label}**",
            f"- score: **{score:.4f}**",
            f"- threshold: {self.cfg.get('threshold', 0.75)}",
            "",
            f"top-{len(topk)} 检索:",
        ]
        for item in topk:
            lines.append(f"- {item['label']}: {item['score']:.4f}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 视频路径解析
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_video_path(video) -> str | None:
        """兼容 gr.Video 返回 str 路径或 dict（含 video/name 键）。"""
        if video is None:
            return None
        if isinstance(video, str):
            return video if Path(video).is_file() else None
        if isinstance(video, dict):
            for key in ("video", "name", "path"):
                p = video.get(key)
                if p and Path(p).is_file():
                    return p
        return None

    # ------------------------------------------------------------------
    # 特征库状态文本
    # ------------------------------------------------------------------
    def list_text(self) -> str:
        names = self.gallery.list()
        if not names:
            return "特征库为空（尚未录入任何类别）。"
        counts = self.gallery.counts()
        lines = [f"共 {len(names)} 个类别，{len(self.gallery)} 条特征：", ""]
        for name in names:
            lines.append(f"- {name}: {counts[name]} 条")
        return "\n".join(lines)

    def gallery_text(self) -> str:
        """视频 Tab 用的特征库状态文本。"""
        names = self.gallery.list()
        if not names:
            return "特征库为空（请先在「录入」Tab 录入类别）。"
        counts = self.gallery.counts()
        lines = [f"共 {len(names)} 个类别，{len(self.gallery)} 条特征：", ""]
        for name in names:
            lines.append(f"- {name}: {counts[name]} 条")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # 摄像头实时识别
    # ------------------------------------------------------------------
    def open_camera(self, index: int) -> tuple[bool, str, np.ndarray | None]:
        """打开摄像头，返回 (是否成功, 状态信息, 首帧画面)。

        用 CAP_DSHOW 在 Windows 下启动更快；失败时返回清晰的错误信息。
        """
        try:
            cap = cv2.VideoCapture(int(index), cv2.CAP_DSHOW)
        except Exception as exc:
            return False, f"❌ 打开摄像头失败：{exc}", None
        if not cap.isOpened():
            cap.release()
            return False, f"❌ 无法打开摄像头 (index={index})，请检查设备是否连接、编号是否正确。", None
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.release()
            return False, f"❌ 摄像头已打开但读不到画面 (index={index})。", None
        h, w = frame.shape[:2]
        self._cam_cap = cap
        self._cam_index = int(index)
        return True, f"✅ 摄像头已连接 (index={index}, {w}×{h})", frame

    def close_camera(self) -> str:
        """关闭摄像头，返回状态文本。"""
        cap = getattr(self, "_cam_cap", None)
        if cap is not None:
            cap.release()
            self._cam_cap = None
        return "🔴 摄像头已关闭"

    def read_frame(self) -> np.ndarray | None:
        """读一帧，失败返回 None。"""
        cap = getattr(self, "_cam_cap", None)
        if cap is None:
            return None
        ok, frame = cap.read()
        if not ok or frame is None:
            return None
        return frame

    @staticmethod
    def draw_guide_box(frame_bgr: np.ndarray) -> np.ndarray:
        """在画面中间画一个参考取景框，提示用户把物体放在框里。

        框尺寸为画面短边的 60%，居中放置；绿色虚线。
        坐标计算与 core.webcam.get_guide_box_rect 保持一致。
        """
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = get_guide_box_rect(h, w)
        side = x2 - x1
        # 画矩形（绿色，2px）
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (70, 200, 70), 2)
        # 四角加装饰短线，更有"取景框"感
        corner = int(side * 0.12)
        color = (70, 200, 70)
        thick = 3
        # 左上
        cv2.line(frame_bgr, (x1, y1), (x1 + corner, y1), color, thick)
        cv2.line(frame_bgr, (x1, y1), (x1, y1 + corner), color, thick)
        # 右上
        cv2.line(frame_bgr, (x2, y1), (x2 - corner, y1), color, thick)
        cv2.line(frame_bgr, (x2, y1), (x2, y1 + corner), color, thick)
        # 左下
        cv2.line(frame_bgr, (x1, y2), (x1 + corner, y2), color, thick)
        cv2.line(frame_bgr, (x1, y2), (x1, y2 - corner), color, thick)
        # 右下
        cv2.line(frame_bgr, (x2, y2), (x2 - corner, y2), color, thick)
        cv2.line(frame_bgr, (x2, y2), (x2, y2 - corner), color, thick)
        return frame_bgr

    def recognize_live_frame(self, use_roi: bool = True) -> tuple[str, str, str, float]:
        """对当前摄像头帧做一次识别。

        默认启用 ROI 裁剪：只识别取景框内的区域（剔除背景干扰，显著提升识别率）。
        返回 (结果 Markdown, top-k 明细文本, label, score)。
        摄像头没开时返回错误信息。
        """
        frame = self.read_frame()
        if frame is None:
            return "❌ 摄像头未开启或已断开，请先开启摄像头。", "", "", 0.0
        t0 = time.time()
        try:
            # 关键修复：裁剪到取景框 ROI 再识别，避免背景噪声淹没特征
            input_frame = crop_guide_box_roi(frame) if use_roi else frame
            label, score, topk = recognize_frame(
                self.recognizer, self.extractor, input_frame
            )
        except Exception as exc:
            return f"❌ 识别失败：{exc}", "", "", 0.0
        elapsed = (time.time() - t0) * 1000
        roi_hint = "（已启用取景框 ROI 裁剪）" if use_roi else "（整帧识别）"
        lines = [
            "**识别结果**",
            "",
            f"- label: **{label}**",
            f"- score: **{score:.4f}**",
            f"- threshold: {self.cfg.get('threshold', 0.75)}",
            f"- 耗时: {elapsed:.0f} ms {roi_hint}",
            "",
            f"top-{len(topk)} 检索:",
        ]
        for item in topk:
            lines.append(f"- {item['label']}: {item['score']:.4f}")
        result_md = "\n".join(lines)
        topk_lines = [
            f"{it['label']} {it['score']:.4f}"
            for it in topk
        ]
        topk_text = " / ".join(topk_lines)
        return result_md, topk_text, label, float(score)

    def capture_and_recognize_multi_frame(
        self, n_frames: int = 7, interval: float = 0.08
    ) -> tuple[str, str, str, float, list[dict]]:
        """多帧捕捉 + 投票识别（新功能）。

        连续捕捉 n_frames 帧（默认 7 帧，间隔 80ms ≈ 0.56s 总时长），
        每帧裁剪取景框 ROI 后识别，最后用多数投票聚合。
        比单帧识别更稳健，可避免某一帧糊/偏/光照差导致识别失败。

        返回 (result_md, topk_text, label, score, frame_details)：
        frame_details 是每帧的明细列表，供界面展示。
        """
        frame_results: list[tuple[str, float, list]] = []
        frame_details: list[dict] = []
        captured = 0
        for i in range(n_frames):
            frame = self.read_frame()
            if frame is None:
                time.sleep(interval)
                continue
            try:
                roi = crop_guide_box_roi(frame)
                label, score, topk = recognize_frame(
                    self.recognizer, self.extractor, roi
                )
                frame_results.append((label, score, topk))
                frame_details.append({
                    "idx": i + 1,
                    "label": label,
                    "score": float(score),
                })
                captured += 1
            except Exception as exc:
                frame_details.append({
                    "idx": i + 1,
                    "label": f"error:{exc}",
                    "score": 0.0,
                })
            time.sleep(interval)

        if captured == 0:
            return (
                "❌ 多帧捕捉失败：摄像头无法读到足够画面。",
                "",
                "",
                0.0,
                frame_details,
            )

        # 复用 core.video 里的多数投票聚合逻辑
        label, score, votes = aggregate_results(
            frame_results,
            unknown_label=self.cfg.get("unknown_label", "unknown"),
        )
        votes_text = "，".join(
            f"{l} {n}帧"
            for l, n in sorted(votes.items(), key=lambda kv: -kv[1])
        )
        lines = [
            f"**多帧识别结果**（{captured}/{n_frames} 帧有效）",
            "",
            f"- label: **{label}**",
            f"- score: **{score:.4f}**（投票获胜类各帧均值）",
            f"- 投票分布: {votes_text}",
            "",
            "逐帧明细:",
        ]
        for d in frame_details:
            lines.append(f"  帧{d['idx']}: {d['label']} ({d['score']:.4f})")

        result_md = "\n".join(lines)
        topk_lines = [
            f"{it['label']} {it['score']:.4f}"
            for it in frame_results[0][2]  # 用第一帧的 topk 作为代表
        ] if frame_results else []
        topk_text = " / ".join(topk_lines)
        return result_md, topk_text, label, float(score), frame_details


def build_demo(cfg: dict) -> gr.Blocks:
    service = RecognitionService(cfg)

    with gr.Blocks(title="物体识别 Demo") as demo:
        gr.Markdown("# 基于 ResNet50 的物体识别 Demo")
        gr.Markdown(
            f"特征库: `{Path(cfg['data_root']) / cfg.get('feature_dir', 'features')}` ｜ "
            f"threshold={cfg.get('threshold', 0.75)} ｜ topk={cfg.get('topk', 5)} ｜ "
            f"frame_interval={service.frame_interval}s"
        )

        with gr.Tabs() as tabs:
            # ==============================================================
            # 录入 Tab：批量录入（目录路径）+ 单个录入（图片上传）
            # ==============================================================
            with gr.Tab("录入", id="enroll") as tab_enroll:
                # ---- 批量录入 ----
                with gr.Accordion("批量录入（目录路径）", open=True):
                    gr.Markdown(
                        "输入包含类别子文件夹的目录路径，每个子文件夹名作为类别名，"
                        "子文件夹内的图片自动录入该类别。\n\n"
                        "目录结构示例：\n"
                        "```\n"
                        "gallery/\n"
                        "  apple/      <- 类别名\n"
                        "    1.jpg     <- 该类别的图片\n"
                        "    2.jpg\n"
                        "  banana/\n"
                        "    1.jpg\n"
                        "```\n\n"
                        f"支持的图片格式：{', '.join(sorted(RecognitionService._IMAGE_EXTS))}"
                    )
                    enroll_dir = gr.Textbox(
                        label="类别目录路径",
                        placeholder="例如 D:\\lrq\\cv_data5.3\\cv_data\\object_recognition_demo\\gallery",
                    )
                    enroll_batch_btn = gr.Button("批量录入", variant="primary")

                # ---- 单个录入 ----
                with gr.Accordion("单个录入（上传图片）", open=False) as enroll_single_accordion:
                    gr.Markdown(
                        "上传 1~10 张图片（同一物体）+ 输入类别名，录入单个类别。\n\n"
                        "录入只把图片特征写入特征库，不会复制或改动你的图片文件。"
                    )
                    with gr.Row():
                        enroll_image = gr.File(
                            label="上传要录入的图片（一次 1~10 张，同一物体）",
                            file_count="multiple",
                            file_types=["image"],
                            type="filepath",
                        )
                        with gr.Column():
                            enroll_name = gr.Textbox(label="类别名称", placeholder="例如 apple")
                            enroll_single_btn = gr.Button("录入", variant="primary")

                # ---- 批量录入结果（左）+ 当前已注册列表（右），文字居中 ----
                with gr.Row():
                    with gr.Column():
                        enroll_status = gr.Markdown(elem_classes=["center-text"])
                    with gr.Column():
                        gr.Markdown(
                            "### 当前已注册列表", elem_classes=["center-text"],
                        )
                        enroll_list = gr.Markdown(
                            value=service.list_text(),
                            elem_classes=["center-text"],
                        )

                # 已注册列表每 3 秒自动刷新，与特征库实时同步
                enroll_auto_refresh = gr.Timer(value=3, active=True)
                enroll_auto_refresh.tick(
                    fn=lambda: service.list_text(), outputs=[enroll_list],
                )

                enroll_batch_btn.click(
                    fn=service.batch_enroll,
                    inputs=[enroll_dir],
                    outputs=[enroll_status, enroll_list],
                )

            # ==============================================================
            # 识别 Tab：上传图片 -> 整图识别 -> 名称 + 相似度
            # ==============================================================
            with gr.Tab("识别", id="recognize") as tab_recognize:
                with gr.Row():
                    rec_image = gr.Image(label="上传要识别的图片", type="filepath")
                    rec_annotated = gr.Image(label="识别结果（原图）")

                with gr.Row():
                    rec_result = gr.Markdown()
                    # ---- 语音播报模块（浏览器端 Web Speech API，纯离线）----
                    with gr.Accordion("🔊 语音播报", open=True):
                        with gr.Row():
                            tts_enabled = gr.Checkbox(
                                label="自动播报识别结果", value=True,
                                info="识别完成后自动朗读结果",
                            )
                            tts_rate = gr.Slider(
                                minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                                label="语速",
                                info="1.0 为正常速度",
                            )
                        tts_play_btn = gr.Button("🔊 手动播放结果", size="sm")
                        tts_status = gr.Markdown(
                            "*使用浏览器自带语音合成，无需联网，在当前设备发声*"
                        )

                # ---- unknown 提示区（识别为 unknown 时显示）----
                with gr.Group(visible=False) as unknown_prompt:
                    unknown_prompt_md = gr.Markdown()
                    with gr.Row():
                        goto_enroll_btn = gr.Button("去录入这张图片", variant="primary")
                        ignore_prompt_btn = gr.Button("忽略")
                # 标记本次录入是否由 unknown 提示发起，决定录入成功后是否自动切回识别页
                from_unknown = gr.State(False)

                # 隐藏文本框：存放要播报的简洁文本，变化时触发 JS 端播报
                tts_text = gr.Textbox(visible=False, label="播报文本")
                rec_btn = gr.Button("识别", variant="primary")

                rec_btn.click(
                    fn=service.detect_and_recognize,
                    inputs=[rec_image],
                    outputs=[
                        rec_annotated, rec_result, tts_text,
                        unknown_prompt, unknown_prompt_md,
                    ],
                )

                def _go_enroll(image_path):
                    if not image_path or not Path(image_path).is_file():
                        return (
                            gr.update(), gr.update(), gr.update(),
                            gr.update(), gr.update(visible=False), False,
                        )
                    return (
                        gr.Tabs(selected="enroll"),
                        gr.update(open=True),
                        gr.update(value=[image_path]),
                        "已带入刚才未能识别的图片。请填写类别名称后点击「录入」，"
                        "录入成功后会自动回到识别页重新识别。",
                        gr.update(visible=False),
                        True,
                    )

                goto_enroll_btn.click(
                    fn=_go_enroll,
                    inputs=[rec_image],
                    outputs=[
                        tabs, enroll_single_accordion, enroll_image,
                        enroll_status, unknown_prompt, from_unknown,
                    ],
                )

                def _dismiss_unknown_prompt():
                    return gr.update(visible=False), "", False

                ignore_prompt_btn.click(
                    fn=_dismiss_unknown_prompt,
                    outputs=[unknown_prompt, unknown_prompt_md, from_unknown],
                )
                rec_image.change(
                    fn=_dismiss_unknown_prompt,
                    outputs=[unknown_prompt, unknown_prompt_md, from_unknown],
                )

                # --- 浏览器端语音播报（Web Speech API）---
                _tts_js = """
                (text, rate, enabled) => {
                    // 参数顺序与 inputs 一致：[tts_text, tts_rate, tts_enabled]
                    if (!enabled || !text) return;
                    if (!('speechSynthesis' in window)) {
                        console.warn('当前浏览器不支持语音合成');
                        return;
                    }
                    // 去掉时间戳前缀（格式：1234567890|实际播报内容）
                    const realText = text.indexOf('|') >= 0 ? text.split('|').slice(1).join('|') : text;
                    if (!realText) return;
                    // 取消之前正在播放的语音，避免堆叠
                    window.speechSynthesis.cancel();
                    const utter = new SpeechSynthesisUtterance(realText);
                    utter.lang = 'zh-CN';
                    utter.rate = parseFloat(rate) || 1.0;
                    utter.pitch = 1.0;
                    utter.volume = 1.0;
                    window.speechSynthesis.speak(utter);
                }
                """
                # 文本变化时自动播报（如果开关打开）
                tts_text.change(
                    fn=None,
                    inputs=[tts_text, tts_rate, tts_enabled],
                    outputs=[],
                    js=_tts_js,
                )
                # 手动播放按钮
                tts_play_btn.click(
                    fn=None,
                    inputs=[tts_text, tts_rate, gr.State(True)],  # 手动播放强制 enabled=True
                    outputs=[],
                    js=_tts_js,
                )

            # ==============================================================
            # 视频识别 Tab：逐帧整帧识别 + 多数投票画廊
            # ==============================================================
            with gr.Tab("视频识别", id="video") as tab_video:
                gr.Markdown(
                    "**说明**：上传视频后按固定间隔抽帧，逐帧对整张帧做检索识别，"
                    "最终通过多数投票得到整段结果。"
                )
                with gr.Row():
                    rec_video = gr.Video(label="上传要识别的视频")
                    with gr.Column():
                        rec_video_result = gr.Markdown()
                        gr.Markdown("### 逐帧明细")
                        rec_video_frames = gr.Dataframe(
                            headers=["时间(s)", "label", "score"],
                            label="逐帧识别结果",
                            interactive=False,
                        )
                rec_video_btn = gr.Button("识别", variant="primary")

                # 逐帧识别结果画廊
                gr.Markdown("### 逐帧识别结果")
                rec_video_gallery = gr.Gallery(
                    label="逐帧图片",
                    columns=3,
                    height=400,
                )

                rec_video_btn.click(
                    fn=service.detect_and_recognize_video,
                    inputs=[rec_video],
                    outputs=[rec_video_result, rec_video_frames, rec_video_gallery],
                )

                gr.Markdown("### 当前特征库（与图片识别共用，录入/管理请在其他 Tab 操作）")
                video_gallery_info = gr.Markdown(value=service.gallery_text())
                video_refresh_btn = gr.Button("刷新特征库状态")

                def _refresh_video_gallery() -> str:
                    # 允许图片页录入后在此热加载最新特征库
                    with service.gallery_lock:
                        if service.gallery.feature_path.exists() and service.gallery.labels_path.exists():
                            service.gallery.load()
                    return service.gallery_text()

                video_refresh_btn.click(fn=_refresh_video_gallery, outputs=[video_gallery_info])

                # 每 3 秒自动刷新特征库状态，与管理页保持一致
                video_auto_refresh_timer = gr.Timer(value=3, active=True)
                video_auto_refresh_timer.tick(fn=_refresh_video_gallery, outputs=[video_gallery_info])

            # ==============================================================
            # 视频实时识别 Tab：沉浸式单按钮体验
            # ==============================================================
            with gr.Tab("视频实时识别", id="live") as tab_live:
                gr.Markdown(
                    "**说明**：调用 USB 摄像头实时显示画面，把物体放在绿色取景框内，"
                    "点击「这是什么」开始 15 秒沉浸式识别。\n\n"
                    "识别过程中可以缓慢转动物体，系统会逐帧投票给出最终结果。"
                )

                # ---- 顶部控制区 ----
                with gr.Row():
                    live_cam_open_btn = gr.Button(
                        "开启摄像头", variant="primary", scale=1,
                    )
                    live_cam_close_btn = gr.Button(
                        "关闭摄像头", variant="stop", scale=1,
                    )
                    live_cam_status = gr.Markdown(
                        "🔴 摄像头未连接", scale=4,
                    )

                # ---- 主区域：画面（左）+ 结果面板（右）----
                with gr.Row():
                    # 左侧：画面 + 大按钮 + 代表帧画廊
                    with gr.Column(scale=2):
                        live_image = gr.Image(
                            label="实时画面",
                            type="numpy",
                            height=480,
                        )
                        live_what_btn = gr.Button(
                            "这是什么", variant="primary", size="lg",
                        )
                        # ---- 学习 / 纠错：仅在本轮识别结束后按结果显示 ----
                        with gr.Row():
                            live_learn_btn = gr.Button(
                                "学习", variant="secondary", visible=False,
                            )
                            live_correct_btn = gr.Button(
                                "纠错", variant="secondary", visible=False,
                            )
                        with gr.Group(visible=False) as live_learn_panel:
                            live_learn_title = gr.Markdown("### 学习")
                            live_learn_gallery = gr.Gallery(
                                label="本轮测试图片（图片名 = 该帧识别结果）",
                                columns=5,
                                height=240,
                            )
                            live_learn_select = gr.CheckboxGroup(
                                label="勾选要录入的图片（可多选）",
                                choices=[],
                            )
                            live_learn_name = gr.Textbox(
                                label="物品名（录入的类别名称）",
                                placeholder="例如 cup",
                            )
                            with gr.Row():
                                live_learn_submit = gr.Button(
                                    "录入特征库", variant="primary",
                                )
                                live_learn_close = gr.Button("收起")
                            live_learn_status = gr.Markdown()
                        gr.Markdown("### 代表帧（获胜类最高/最低分）")
                        with gr.Row():
                            live_frame_high = gr.Image(
                                label="最高分",
                                type="numpy",
                                height=240,
                            )
                            live_frame_low = gr.Image(
                                label="最低分",
                                type="numpy",
                                height=240,
                            )

                    # 右侧：投票排行榜 + 逐帧明细
                    with gr.Column(scale=1):
                        live_rank_md = gr.Markdown(
                            "📊 **投票排行榜**\n\n"
                            "第 0 / 30 帧\n\n"
                            "（点击「这是什么」开始识别）"
                        )
                        with gr.Accordion("逐帧明细", open=False):
                            live_detail_md = gr.Markdown("（暂无数据）")
                        # 特征库状态（3 秒自动刷新）
                        gr.Markdown("---")
                        live_gallery_info = gr.Markdown(value=service.gallery_text())
                        live_refresh_btn = gr.Button("刷新特征库", size="sm")

                def _refresh_live_gallery() -> str:
                    with service.gallery_lock:
                        if service.gallery.feature_path.exists() and service.gallery.labels_path.exists():
                            service.gallery.load()
                    return service.gallery_text()

                live_refresh_btn.click(fn=_refresh_live_gallery, outputs=[live_gallery_info])
                live_auto_refresh = gr.Timer(value=3, active=True)
                live_auto_refresh.tick(fn=_refresh_live_gallery, outputs=[live_gallery_info])

                # ---- 隐藏播报文本框 ----
                live_tts_text = gr.Textbox(visible=False, label="播报文本")

                # ---- 状态变量 ----
                # live_state: "cam_off" / "idle" / "countdown_down" / "recognizing" / "countdown_up"
                live_state = gr.State("cam_off")
                # 阶段开始时间戳（用于倒计时/识别计时）
                live_phase_start = gr.State(0.0)
                # 上一次识别的帧索引（避免 50ms 内重复识别）
                live_last_frame_idx = gr.State(-1)
                # 投票数据：dict[label] = [score1, score2, ...]
                live_scores = gr.State({})
                # 逐帧明细：list of dict {t, label, score}
                live_details = gr.State([])
                # 逐帧画面缓存：list of numpy RGB 数组（和 details 一一对应）
                live_frames_rgb = gr.State([])
                # 逐帧取景框 ROI 干净图缓存（无字幕/无绿框，与 details 一一对应，
                # 供「学习 / 纠错」面板勾选录入用）
                live_frames_roi = gr.State([])
                # 最终结果停留：(winners_list, score_float, is_unknown_bool)
                live_final_result = gr.State(("", 0.0, True))
                # 倒计时已播报的字索引（避免重复播报）
                live_countdown_spoken = gr.State(-1)

                # ---- 画面刷新定时器（单 Timer 驱动所有逻辑，20 FPS）----
                live_frame_timer = gr.Timer(value=0.05, active=False)

                # ---- 常量 ----
                RECOGNIZE_DURATION = 15.0    # 识别总时长（秒）
                RECOGNIZE_INTERVAL = 0.5     # 识别间隔（秒）
                TOTAL_FRAMES = 30            # 总帧数
                COUNTDOWN_STEPS = ["3", "2", "1", "开始识别"]  # 预热倒计时
                COUNTDOWN_UP_STEPS = ["3", "2", "1", "结束识别"]  # 结束倒计时
                UNKNOWN_LABEL = cfg.get("unknown_label", "unknown")

                def _compute_winners(scores_by_label: dict) -> tuple[list[str], float, bool]:
                    """根据置信度加权投票（sum 规则）计算获胜者。

                    返回 (winner_labels, avg_score, is_unknown)：
                    - winner_labels: 并列第一的类别名列表（按名称排序）
                    - avg_score: 获胜类别的平均分（sum / 票数）
                    - is_unknown: 获胜者是否为 unknown
                    """
                    if not scores_by_label:
                        return [UNKNOWN_LABEL], 0.0, True

                    # 计算每类总分（sum 规则）
                    total_by_label = {
                        label: sum(scores)
                        for label, scores in scores_by_label.items()
                    }
                    max_total = max(total_by_label.values())
                    # 并列第一：差值 <= 0.001
                    winners = [
                        label for label, total in total_by_label.items()
                        if max_total - total <= 0.001
                    ]
                    winners.sort()  # 按名称排序

                    # 获胜类别的平均分（用第一个 winner 计算，分数接近）
                    first_winner = winners[0]
                    scores_list = scores_by_label[first_winner]
                    avg = sum(scores_list) / len(scores_list) if scores_list else 0.0

                    is_unk = (len(winners) == 1 and winners[0] == UNKNOWN_LABEL)
                    return winners, avg, is_unk

                def _build_rank_md(
                    scores_by_label: dict,
                    frame_idx: int,
                    total_frames: int = TOTAL_FRAMES,
                    result: tuple | None = None,
                ) -> str:
                    """构建投票排行榜 Markdown（带 █ 进度条）。

                    result 为 (winners, avg_score, is_unknown) 时，在排行榜容器内
                    追加「识别结果 / 相似度」两行；为 None（尚无结果）则不显示。
                    """
                    if not scores_by_label:
                        return (
                            f"📊 **投票排行榜**\n\n"
                            f"第 {frame_idx} / {total_frames} 帧\n\n"
                            "（等待识别开始...）"
                        )

                    # 计算每类总分 + 票数
                    stats = []
                    for label, scores in scores_by_label.items():
                        total = sum(scores)
                        count = len(scores)
                        avg = total / count if count else 0.0
                        stats.append((label, count, total, avg))

                    # 按总分降序
                    stats.sort(key=lambda x: -x[2])
                    max_total = max(s[2] for s in stats) if stats else 1.0

                    lines = [
                        f"📊 **投票排行榜**",
                        "",
                        f"第 {frame_idx} / {total_frames} 帧",
                        "",
                    ]
                    if result is not None:
                        winners, avg, is_unk = result
                        if is_unk:
                            result_disp = UNKNOWN_LABEL
                        elif len(winners) == 1:
                            result_disp = winners[0]
                        else:
                            result_disp = " / ".join(winners)
                        lines.append(f"识别结果：{result_disp}")
                        lines.append(f"相似度：{avg:.4f}")
                        lines.append("")
                    bar_len = 15
                    for label, count, total, avg in stats:
                        filled = int(bar_len * total / max_total) if max_total > 0 else 0
                        bar = "█" * filled + "." * (bar_len - filled)
                        label_disp = label if len(label) <= 6 else label[:6]
                        lines.append(
                            f"`{label_disp:<6}`  {count:>2}票  "
                            f"{bar}  {avg:.2f}"
                        )
                    return "\n".join(lines)

                def _build_detail_md(details: list[dict]) -> str:
                    """构建逐帧明细 Markdown（表格形式，三列对齐）。"""
                    if not details:
                        return "（暂无数据）"
                    lines = [
                        "| 时间 | 类别 | 相似度 |",
                        "|------|------|--------|",
                    ]
                    for d in details:
                        lines.append(
                            f"| {d['t']:.1f}s | {d['label']} | {d['score']:.2f} |"
                        )
                    return "\n".join(lines)

                def _get_winner_frames(
                    winners: list[str],
                    details: list[dict],
                    frames_rgb: list,
                ) -> tuple:
                    """从获胜类的帧中选出最高/最低分帧。

                    返回 (high_frame_rgb, low_frame_rgb)，没有则返回 (None, None)
                    """
                    if not winners or not details or not frames_rgb:
                        return None, None
                    winner_label = winners[0]
                    winner_frames = [
                        (i, d["score"])
                        for i, d in enumerate(details)
                        if d["label"] == winner_label and i < len(frames_rgb)
                    ]
                    if not winner_frames:
                        return None, None
                    winner_frames.sort(key=lambda x: x[1])
                    lowest_idx, _ = winner_frames[0]
                    highest_idx, _ = winner_frames[-1]
                    high_frame = frames_rgb[highest_idx]
                    low_frame = (
                        frames_rgb[lowest_idx]
                        if highest_idx != lowest_idx
                        else None
                    )
                    return high_frame, low_frame

                def _make_tts(text: str) -> str:
                    """生成带时间戳的播报文本。"""
                    ts = str(int(time.time() * 1000))
                    return f"{ts}|{text}"

                def _speak_winners(winners: list[str], score: float, is_unknown: bool) -> str:
                    """生成最终结果的语音播报文本。"""
                    if is_unknown:
                        return "未能识别出已知物体"
                    pct = int(score * 100)
                    if len(winners) == 1:
                        return f"识别结果，这可能是{winners[0]}，相似度百分之{pct}"
                    else:
                        names = "、".join(winners[:-1]) + "和" + winners[-1]
                        return f"识别结果，{names}并列第一，相似度百分之{pct}"

                def _subtitle_for_winners(
                    winners: list[str], score: float, is_unknown: bool,
                ) -> tuple[str, str, tuple]:
                    """生成最终结果的字幕（大字、小字、颜色）。"""
                    if is_unknown:
                        return "未能识别出已知物体", "", COLOR_RED
                    if len(winners) == 1:
                        return f"{winners[0]}  {score:.2f}", "相似度", COLOR_GREEN
                    else:
                        label_str = " / ".join(winners)
                        return f"{label_str}  并列", f"{score:.2f}", COLOR_GREEN

                def _tick_live(
                    state: str,
                    phase_start: float,
                    last_frame_idx: int,
                    scores_by_label: dict,
                    details: list,
                    frames_rgb: list,
                    frames_roi: list,
                    final_result: tuple,
                    countdown_spoken: int,
                ):
                    """单 Timer 驱动的主循环（每 50ms 一次）。

                    根据当前状态 + 经过的时间，决定该干嘛：
                    - 读帧 + 画取景框 + 画字幕
                    - 倒计时逐字播报
                    - 每 0.5s 识别一帧 + 更新投票
                    - 状态切换
                    """
                    # ---- 读帧 ----
                    frame = service.read_frame()
                    if frame is None:
                        # 摄像头断开
                        if state != "cam_off":
                            service.close_camera()
                            tts_val = _make_tts("摄像头已断开")
                            rank_md_empty = (
                                "📊 **投票排行榜**\n\n"
                                "第 0 / 30 帧\n\n"
                                "（点击「这是什么」开始识别）"
                            )
                            return (
                                None,          # image
                                "🔴 摄像头已断开",  # status
                                "cam_off",     # state
                                0.0,           # phase_start
                                -1,            # last_frame_idx
                                {},            # scores
                                [],            # details
                                [],            # frames_rgb
                                [],            # frames_roi
                                ("", 0.0, True),  # final_result
                                -1,            # countdown_spoken
                                gr.update(interactive=False),  # what_btn
                                gr.update(visible=False),      # learn_btn
                                gr.update(visible=False),      # correct_btn
                                tts_val,       # tts
                                rank_md_empty,  # rank_md
                                "（暂无数据）",  # detail_md
                                None,          # frame_high
                                None,          # frame_low
                            )
                        rank_md_empty = (
                            "📊 **投票排行榜**\n\n"
                            "第 0 / 30 帧\n\n"
                            "（点击「这是什么」开始识别）"
                        )
                        return (
                            None, "🔴 摄像头未连接", "cam_off", 0.0, -1,
                            {}, [], [], [], ("", 0.0, True), -1,
                            gr.update(interactive=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            "",
                            rank_md_empty, "（暂无数据）", None, None,
                        )

                    now = time.time()
                    elapsed = now - phase_start
                    big_text = ""
                    small_text = ""
                    color = COLOR_GREEN
                    tts_out = ""
                    new_state = state
                    new_phase_start = phase_start
                    new_scores = scores_by_label
                    new_details = details
                    new_frames_rgb = frames_rgb
                    new_final = final_result
                    new_countdown_spoken = countdown_spoken
                    new_frame_idx = last_frame_idx
                    new_frames_roi = frames_roi
                    new_frame_high = None
                    new_frame_low = None
                    _new_frame_captured = False
                    roi_rgb = None  # 本帧取景框 ROI 干净图（供学习/纠错录入）
                    what_btn_interactive = True

                    # ---- 按状态处理 ----
                    if state == "idle":
                        # 待命状态：显示实时画面 + 上一次结果
                        if final_result[0]:
                            big_text, small_text, color = _subtitle_for_winners(
                                *final_result,
                            )
                            # 保持代表帧显示（从缓存重建）
                            new_frame_high, new_frame_low = _get_winner_frames(
                                final_result[0], details, frames_rgb,
                            )
                        else:
                            big_text = "准备就绪"
                            small_text = "按「这是什么」开始识别"
                            color = COLOR_GREEN
                        what_btn_interactive = True

                    elif state == "countdown_down":
                        # 预热倒计时：3→2→1→开始识别，每字一秒
                        step_idx = int(elapsed)
                        if step_idx >= len(COUNTDOWN_STEPS):
                            # 进入识别阶段
                            new_state = "recognizing"
                            new_phase_start = now
                            new_scores = {}
                            new_details = []
                            new_frame_idx = -1
                            big_text = "正在识别中... 剩余 15s"
                            small_text = "（可以转动物体）"
                            color = COLOR_GREEN
                        else:
                            big_text = COUNTDOWN_STEPS[step_idx]
                            small_text = ""
                            color = COLOR_GREEN
                            # 到新字时播报
                            if step_idx > countdown_spoken:
                                new_countdown_spoken = step_idx
                                tts_out = _make_tts(COUNTDOWN_STEPS[step_idx])
                        what_btn_interactive = False

                    elif state == "recognizing":
                        # 识别中：15 秒，每 0.5s 一帧
                        remaining = max(0, int(RECOGNIZE_DURATION - elapsed))
                        big_text = f"正在识别中... 剩余 {remaining}s"
                        small_text = "（可以转动物体）"
                        color = COLOR_GREEN
                        what_btn_interactive = False

                        # 计算当前应该是第几帧
                        current_frame_idx = int(elapsed / RECOGNIZE_INTERVAL)
                        if current_frame_idx >= TOTAL_FRAMES:
                            # 识别结束，进入结束倒计时
                            new_state = "countdown_up"
                            new_phase_start = now
                            new_countdown_spoken = -1
                            big_text = COUNTDOWN_UP_STEPS[0]
                            small_text = ""
                        elif current_frame_idx > last_frame_idx:
                            # 该做新的一帧识别了
                            try:
                                roi = crop_guide_box_roi(frame)
                                # 先保存干净 ROI（无字幕/无绿框），再识别
                                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                                label, score, _topk = recognize_frame(
                                    service.recognizer, service.extractor, roi,
                                )
                            except Exception:
                                label = UNKNOWN_LABEL
                                score = 0.0

                            # unknown 也要记录，score 用该帧最高相似度（即使低于阈值）
                            # 注意：recognize_frame 返回的 label 如果是 unknown，
                            # score 就是该帧最高相似度（recognizer.predict_feature 的行为）
                            if label not in new_scores:
                                new_scores[label] = []
                            new_scores[label].append(float(score))

                            t_val = current_frame_idx * RECOGNIZE_INTERVAL
                            new_details = new_details + [
                                {"t": t_val, "label": label, "score": float(score)}
                            ]
                            new_frame_idx = current_frame_idx
                            _new_frame_captured = True
                        else:
                            # 还没到下一帧，保持
                            pass

                    elif state == "countdown_up":
                        # 结束倒计时：3→2→1→结束识别，每字一秒
                        step_idx = int(elapsed)
                        if step_idx >= len(COUNTDOWN_UP_STEPS):
                            # 出最终结果，回到 IDLE
                            winners, avg, is_unk = _compute_winners(scores_by_label)
                            new_final = (winners, avg, is_unk)
                            new_state = "idle"
                            big_text, small_text, color = _subtitle_for_winners(
                                winners, avg, is_unk,
                            )
                            tts_out = _make_tts(_speak_winners(winners, avg, is_unk))
                            what_btn_interactive = True
                            # 组装获胜类代表帧
                            new_frame_high, new_frame_low = _get_winner_frames(
                                winners, new_details, new_frames_rgb,
                            )
                        else:
                            big_text = COUNTDOWN_UP_STEPS[step_idx]
                            small_text = ""
                            color = COLOR_GREEN
                            if step_idx > countdown_spoken:
                                new_countdown_spoken = step_idx
                                tts_out = _make_tts(COUNTDOWN_UP_STEPS[step_idx])
                            what_btn_interactive = False

                    # ---- 画取景框 + 字幕 ----
                    service.draw_guide_box(frame)
                    draw_subtitle_bar(frame, big_text, small_text, color)
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    # ---- 存帧（识别新帧时存一份完整 RGB 快照 + 干净 ROI）----
                    if _new_frame_captured:
                        new_frames_rgb = new_frames_rgb + [rgb.copy()]
                        new_frames_roi = frames_roi + [
                            roi_rgb if roi_rgb is not None else rgb.copy(),
                        ]

                    # ---- 学习 / 纠错按钮可见性 ----
                    # 仅当一轮识别结束回到 idle 且有最终结果时显示：
                    # 结果为 unknown → 显示「学习」；结果为已知类别 → 显示「纠错」；
                    # 点击「这是什么」执行期间（倒计时/识别中）→ 两者都隐藏。
                    if new_state == "idle" and new_final[0]:
                        learn_vis = gr.update(visible=bool(new_final[2]))
                        correct_vis = gr.update(visible=not new_final[2])
                    else:
                        learn_vis = gr.update(visible=False)
                        correct_vis = gr.update(visible=False)

                    # ---- 状态文字 ----
                    status_text = "🟢 摄像头已连接"

                    # ---- 排行榜 + 明细 ----
                    # 实时更新（识别中才更新数据，否则保持上一次）
                    if state == "recognizing":
                        interim = _compute_winners(new_scores) if new_scores else None
                        rank_md = _build_rank_md(
                            new_scores, new_frame_idx + 1, result=interim,
                        )
                        detail_md = _build_detail_md(new_details)
                    elif state in ("countdown_down", "countdown_up"):
                        interim = (
                            _compute_winners(scores_by_label)
                            if scores_by_label else None
                        )
                        rank_md = _build_rank_md(
                            scores_by_label, len(details), result=interim,
                        )
                        detail_md = _build_detail_md(details)
                    else:
                        # idle/cam_off：显示最终结果的排行榜（如果有）
                        if scores_by_label:
                            rank_md = _build_rank_md(
                                scores_by_label, TOTAL_FRAMES,
                                result=final_result if final_result[0] else None,
                            )
                            detail_md = _build_detail_md(details)
                        else:
                            rank_md = (
                                "📊 **投票排行榜**\n\n"
                                "第 0 / 30 帧\n\n"
                                "（点击「这是什么」开始识别）"
                            )
                            detail_md = "（暂无数据）"

                    return (
                        rgb,                       # image
                        status_text,               # status
                        new_state,                 # state
                        new_phase_start,           # phase_start
                        new_frame_idx,             # last_frame_idx
                        new_scores,                # scores
                        new_details,               # details
                        new_frames_rgb,            # frames_rgb
                        new_frames_roi,            # frames_roi
                        new_final,                 # final_result
                        new_countdown_spoken,      # countdown_spoken
                        gr.update(interactive=what_btn_interactive),  # what_btn
                        learn_vis,                 # learn_btn
                        correct_vis,               # correct_btn
                        tts_out,                   # tts
                        rank_md,                   # rank_md
                        detail_md,                 # detail_md
                        new_frame_high,            # frame_high
                        new_frame_low,             # frame_low
                    )

                live_frame_timer.tick(
                    fn=_tick_live,
                    inputs=[
                        live_state,
                        live_phase_start,
                        live_last_frame_idx,
                        live_scores,
                        live_details,
                        live_frames_rgb,
                        live_frames_roi,
                        live_final_result,
                        live_countdown_spoken,
                    ],
                    outputs=[
                        live_image,
                        live_cam_status,
                        live_state,
                        live_phase_start,
                        live_last_frame_idx,
                        live_scores,
                        live_details,
                        live_frames_rgb,
                        live_frames_roi,
                        live_final_result,
                        live_countdown_spoken,
                        live_what_btn,
                        live_learn_btn,
                        live_correct_btn,
                        live_tts_text,
                        live_rank_md,
                        live_detail_md,
                        live_frame_high,
                        live_frame_low,
                    ],
                )

                # ---- 开启摄像头 ----
                def _live_open_camera():
                    ok, msg, frame = service.open_camera(0)  # 默认 0 号
                    if not ok or frame is None:
                        return (
                            f"🔴 {msg}", None, "cam_off", 0.0, -1,
                            {}, [], [], [], ("", 0.0, True), -1,
                            gr.update(active=False),
                            gr.update(interactive=False),
                            gr.update(visible=False),
                            gr.update(visible=False),
                            "",
                            None, None,
                            gr.update(visible=False),
                        )
                    service.draw_guide_box(frame)
                    draw_subtitle_bar(
                        frame, "准备就绪", "按「这是什么」开始识别", COLOR_GREEN,
                    )
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    tts_val = _make_tts("摄像头已开启")
                    return (
                        "🟢 摄像头已连接", rgb, "idle", 0.0, -1,
                        {}, [], [], [], ("", 0.0, True), -1,
                        gr.update(active=True),
                        gr.update(interactive=True),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        tts_val,
                        None, None,
                        gr.update(visible=False),
                    )

                live_cam_open_btn.click(
                    fn=_live_open_camera,
                    outputs=[
                        live_cam_status, live_image, live_state, live_phase_start,
                        live_last_frame_idx, live_scores, live_details,
                        live_frames_rgb, live_frames_roi,
                        live_final_result, live_countdown_spoken,
                        live_frame_timer, live_what_btn,
                        live_learn_btn, live_correct_btn,
                        live_tts_text,
                        live_frame_high, live_frame_low,
                        live_learn_panel,
                    ],
                )

                # ---- 关闭摄像头 ----
                def _live_close_camera(state):
                    msg = service.close_camera()
                    tts_val = _make_tts("摄像头已关闭")
                    return (
                        "🔴 摄像头已关闭", None, "cam_off", 0.0, -1,
                        {}, [], [], [], ("", 0.0, True), -1,
                        gr.update(active=False),
                        gr.update(interactive=False),
                        gr.update(visible=False),
                        gr.update(visible=False),
                        tts_val,
                        None, None,
                        gr.update(visible=False),
                    )

                live_cam_close_btn.click(
                    fn=_live_close_camera,
                    inputs=[live_state],
                    outputs=[
                        live_cam_status, live_image, live_state, live_phase_start,
                        live_last_frame_idx, live_scores, live_details,
                        live_frames_rgb, live_frames_roi,
                        live_final_result, live_countdown_spoken,
                        live_frame_timer, live_what_btn,
                        live_learn_btn, live_correct_btn,
                        live_tts_text,
                        live_frame_high, live_frame_low,
                        live_learn_panel,
                    ],
                )

                # ---- 点击「这是什么」----
                def _live_what_is_this(state):
                    if state != "idle":
                        # 只能在 IDLE 状态下点
                        return (
                            state, time.time(), -1, {}, [], [], [],
                            ("", 0.0, True), -1,
                            gr.update(interactive=False),
                            "",
                            None, None,
                            gr.update(visible=False),
                        )
                    # 进入预热倒计时（同时收起学习/纠错面板）
                    return (
                        "countdown_down",  # state
                        time.time(),       # phase_start
                        -1,                # last_frame_idx
                        {},                # scores（清空）
                        [],                # details（清空）
                        [],                # frames_rgb（清空）
                        [],                # frames_roi（清空）
                        ("", 0.0, True),   # final_result（清空）
                        -1,                # countdown_spoken
                        gr.update(interactive=False),  # 按钮变灰
                        "",                # tts（第一秒的"3"由 tick 里播）
                        None, None,        # frame_high / frame_low（清空）
                        gr.update(visible=False),  # 收起学习/纠错面板
                    )

                live_what_btn.click(
                    fn=_live_what_is_this,
                    inputs=[live_state],
                    outputs=[
                        live_state, live_phase_start, live_last_frame_idx,
                        live_scores, live_details, live_frames_rgb,
                        live_frames_roi,
                        live_final_result, live_countdown_spoken,
                        live_what_btn, live_tts_text,
                        live_frame_high, live_frame_low,
                        live_learn_panel,
                    ],
                )

                # ---- 学习 / 纠错面板 ----
                def _build_learn_items(
                    frames_roi: list, details: list,
                ) -> tuple[list, list[str]]:
                    """把本轮 ROI 帧组装成画廊项与勾选项。

                    caption 即"图片名"：帧序号 + 该帧识别结果（label + 相似度），
                    帧序号保证唯一，可与勾选值一一对应。
                    """
                    items: list = []
                    for i, (roi, d) in enumerate(zip(frames_roi, details)):
                        caption = f"帧{i + 1}: {d['label']} ({d['score']:.2f})"
                        items.append((roi, caption))
                    return items, [c for _, c in items]

                def _open_learn_panel(mode: str, frames_roi, details):
                    """打开学习/纠错面板：展示本轮测试图片，供多选录入。"""
                    items, captions = _build_learn_items(frames_roi, details)
                    title = (
                        "### 学习：把本轮测试图片录入为新的物品"
                        if mode == "learn"
                        else "### 纠错：把本轮测试图片录入为正确物品"
                    )
                    status = (
                        "*勾选图片 → 填写物品名 → 点击「录入特征库」，"
                        "一轮可多次录入。*"
                        if items
                        else "（本轮没有可用的测试图片，无法录入）"
                    )
                    return (
                        gr.update(visible=True),
                        gr.update(value=title),
                        gr.update(value=items),
                        gr.update(choices=captions, value=[]),
                        gr.update(value=""),
                        gr.update(value=status),
                    )

                live_learn_btn.click(
                    fn=lambda roi, det: _open_learn_panel("learn", roi, det),
                    inputs=[live_frames_roi, live_details],
                    outputs=[
                        live_learn_panel, live_learn_title, live_learn_gallery,
                        live_learn_select, live_learn_name, live_learn_status,
                    ],
                )
                live_correct_btn.click(
                    fn=lambda roi, det: _open_learn_panel("correct", roi, det),
                    inputs=[live_frames_roi, live_details],
                    outputs=[
                        live_learn_panel, live_learn_title, live_learn_gallery,
                        live_learn_select, live_learn_name, live_learn_status,
                    ],
                )
                live_learn_close.click(
                    fn=lambda: gr.update(visible=False),
                    outputs=[live_learn_panel],
                )

                def _submit_learn(selection, name, frames_roi, details):
                    """把勾选的本轮测试图片按单个录入逻辑分批写入特征库。"""
                    selection = selection or []
                    if not selection:
                        return "请先勾选至少一张图片。", gr.update()
                    _items, captions = _build_learn_items(frames_roi, details)
                    idxs = [captions.index(s) for s in selection]
                    # ROI 帧写入临时 PNG，交给单个录入逻辑提特征入库
                    tmpdir = tempfile.mkdtemp(prefix="live_learn_")
                    paths: list[str] = []
                    try:
                        for i in idxs:
                            p = Path(tmpdir) / f"frame_{i + 1:02d}.png"
                            Image.fromarray(frames_roi[i]).save(str(p))
                            paths.append(str(p))
                        ok, msg = service.enroll_paths_chunked(name, paths)
                    finally:
                        shutil.rmtree(tmpdir, ignore_errors=True)
                    if ok:
                        msg += f"（本次共录入 {len(paths)} 张，可继续勾选再次录入）"
                    return msg, gr.update(value=[])

                live_learn_submit.click(
                    fn=_submit_learn,
                    inputs=[
                        live_learn_select, live_learn_name,
                        live_frames_roi, live_details,
                    ],
                    outputs=[live_learn_status, live_learn_select],
                )

                # ---- 语音播报：客户端轮询检测文本变化并播报 ----
                # （Timer 驱动的文本更新不会触发 .change() 事件，所以用轮询代替）
                _live_tts_js = """
                (text) => {
                    if (!text) return;
                    if (!('speechSynthesis' in window)) return;
                    if (!window._live_last_tts) window._live_last_tts = '';
                    if (text === window._live_last_tts) return;
                    window._live_last_tts = text;
                    const realText = text.indexOf('|') >= 0
                        ? text.split('|').slice(1).join('|')
                        : text;
                    if (!realText) return;
                    window.speechSynthesis.cancel();
                    const utter = new SpeechSynthesisUtterance(realText);
                    utter.lang = 'zh-CN';
                    utter.rate = 1.0;
                    utter.pitch = 1.0;
                    utter.volume = 1.0;
                    window.speechSynthesis.speak(utter);
                }
                """
                live_tts_timer = gr.Timer(value=0.2, active=True)
                live_tts_timer.tick(
                    fn=None,
                    inputs=[live_tts_text],
                    outputs=[],
                    js=_live_tts_js,
                )

            with gr.Tab("管理", id="manage") as tab_manage:
                gr.Markdown("### 已注册类别列表")
                manage_list = gr.Markdown(value=service.list_text())

                gr.Markdown("### 删除类别")
                gr.Markdown(
                    "勾选要删除的类别（可多选，即删除一个或多个），或点击「删除全部」清空特征库。\n\n"
                    "删除只移除特征库里的条目，不会改动你本地的任何图片文件。"
                )
                delete_choices = gr.CheckboxGroup(
                    label="选择要删除的类别",
                    choices=service.gallery.list(),
                )
                with gr.Row():
                    delete_selected_btn = gr.Button("删除选中", variant="stop")
                    delete_all_btn = gr.Button("删除全部", variant="stop")
                delete_status = gr.Markdown()
                refresh_btn = gr.Button("刷新列表")
                # 每 3 秒自动刷新列表与勾选项；勾选仍存在的类别会被保留，不打断用户操作。
                auto_refresh_timer = gr.Timer(value=3, active=True)

                def _refresh(selected=None) -> tuple[str, object]:
                    names = service.gallery.list()
                    # 手动刷新不保留勾选；自动刷新保留仍存在的勾选项
                    if selected is None:
                        kept: list[str] = []
                    else:
                        kept = [s for s in (selected or []) if s in names]
                    return service.list_text(), gr.update(choices=names, value=kept)

                refresh_btn.click(fn=_refresh, outputs=[manage_list, delete_choices])
                auto_refresh_timer.tick(
                    fn=_refresh,
                    inputs=[delete_choices],
                    outputs=[manage_list, delete_choices],
                )

                def _delete_selected(
                    selected,
                ) -> tuple[str, str, str, object, object]:
                    selected = selected or []
                    if not selected:
                        return (
                            "未选中任何类别。",
                            service.list_text(),
                            service.list_text(),
                            gr.update(value=[]),
                            gr.update(),
                        )
                    with service.gallery_lock:
                        removed_total = 0
                        for name in selected:
                            removed_total += service.gallery.delete(name)
                        service.gallery.save()
                        remaining = service.gallery.list()
                    status = (
                        f"已删除 {len(selected)} 个类别，共 {removed_total} 条特征，"
                        f"剩余 {len(remaining)} 个类别。"
                    )
                    list_text = service.list_text()
                    return (
                        status,
                        list_text,
                        list_text,
                        gr.update(choices=remaining, value=[]),
                        # 特征库被删空时，同步清空录入页的批量录入结果
                        "" if not remaining else gr.update(),
                    )

                def _delete_all() -> tuple[str, str, str, object, object]:
                    names = service.gallery.list()
                    if not names:
                        return (
                            "特征库已为空，无需删除。",
                            service.list_text(),
                            service.list_text(),
                            gr.update(choices=[], value=[]),
                            gr.update(),
                        )
                    with service.gallery_lock:
                        removed_total = 0
                        for name in names:
                            removed_total += service.gallery.delete(name)
                        service.gallery.save()
                    status = (
                        f"已删除全部 {len(names)} 个类别，共 {removed_total} 条特征，"
                        f"特征库已清空。"
                    )
                    list_text = service.list_text()
                    return (
                        status,
                        list_text,
                        list_text,
                        gr.update(choices=[], value=[]),
                        "",  # 删除全部后清空录入页的批量录入结果
                    )

                delete_selected_btn.click(
                    fn=_delete_selected,
                    inputs=[delete_choices],
                    outputs=[
                        delete_status, manage_list, enroll_list,
                        delete_choices, enroll_status,
                    ],
                )
                delete_all_btn.click(
                    fn=_delete_all,
                    outputs=[
                        delete_status, manage_list, enroll_list,
                        delete_choices, enroll_status,
                    ],
                )

            # ---- Tab 切换监听：切到非 live Tab 时自动关摄像头 ----
            def _live_tab_leave():
                """离开视频实时识别 Tab 时：关闭摄像头、停止识别、画面变黑。"""
                service.close_camera()
                rank_md_empty = (
                    "📊 **投票排行榜**\n\n"
                    "第 0 / 30 帧\n\n"
                    "（点击「这是什么」开始识别）"
                )
                return (
                    "🔴 摄像头未连接",   # status
                    None,                 # image
                    "cam_off",            # state
                    0.0,                  # phase_start
                    -1,                   # last_frame_idx
                    {},                   # scores
                    [],                   # details
                    [],                   # frames_rgb
                    [],                   # frames_roi
                    ("", 0.0, True),      # final_result
                    -1,                   # countdown_spoken
                    gr.update(active=False),  # timer
                    gr.update(interactive=False),  # what_btn
                    gr.update(visible=False),  # learn_btn
                    gr.update(visible=False),  # correct_btn
                    "",                   # tts
                    rank_md_empty,        # rank_md
                    "（暂无数据）",        # detail_md
                    None, None,           # frame_high / frame_low
                    gr.update(visible=False),  # learn_panel
                )

            # 切到其他 4 个 Tab 时都触发关摄像头（输出列表共用）
            leave_outputs = [
                live_cam_status, live_image, live_state, live_phase_start,
                live_last_frame_idx, live_scores, live_details,
                live_frames_rgb, live_frames_roi,
                live_final_result, live_countdown_spoken,
                live_frame_timer, live_what_btn,
                live_learn_btn, live_correct_btn,
                live_tts_text,
                live_rank_md, live_detail_md,
                live_frame_high, live_frame_low,
                live_learn_panel,
            ]
            tab_enroll.select(fn=_live_tab_leave, outputs=leave_outputs)
            tab_recognize.select(fn=_live_tab_leave, outputs=leave_outputs)
            tab_video.select(fn=_live_tab_leave, outputs=leave_outputs)
            tab_manage.select(fn=_live_tab_leave, outputs=leave_outputs)

        # ==================================================================
        # 单个录入的点击回调：放在所有 Tab 建完之后绑定，因为「来自 unknown 提示
        # 流程」的这次录入成功后需要回到识别页，并引用识别页上的组件做重新识别
        # ==================================================================
        def _enroll_single(images, name, came_from_unknown):
            status, list_text, paths = service.enroll(images, name)
            if came_from_unknown and paths:
                image, result_md, tts_value, prompt_update, prompt_md = (
                    service.detect_and_recognize(paths[0])
                )
                return (
                    status, list_text,
                    gr.Tabs(selected="recognize"),
                    image, result_md, tts_value, prompt_update, prompt_md,
                    False,
                )
            return (
                status, list_text,
                gr.update(), gr.update(), gr.update(), gr.update(),
                gr.update(), gr.update(), False,
            )

        enroll_single_btn.click(
            fn=_enroll_single,
            inputs=[enroll_image, enroll_name, from_unknown],
            outputs=[
                enroll_status, enroll_list,
                tabs, rec_annotated, rec_result, tts_text,
                unknown_prompt, unknown_prompt_md, from_unknown,
            ],
        )

    return demo


def main() -> None:
    with open(DEFAULT_CONFIG_PATH, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    demo = build_demo(cfg)
    # Gradio 6 起 css 参数移到 launch()；center-text 用于录入页左右两栏文字居中
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        css=".center-text {text-align: center;}",
    )


if __name__ == "__main__":
    main()
