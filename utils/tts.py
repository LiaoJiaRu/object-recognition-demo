"""utils/tts.py

离线语音播报模块（基于 pyttsx3）。
- 完全离线，不调用任何在线 API
- 全局只初始化一次语音引擎（单例模式），防止重复初始化报错
- 支持中文播报，可调节语速
- 失败时自动降级为文本打印，不影响主流程

安装命令:
    pip install pyttsx3

使用方式:
    from utils.tts import speak, set_rate
    set_rate(180)       # 设置语速（字/分钟），默认 200
    speak("这是一个苹果")  # 播放语音
"""
from __future__ import annotations

import importlib.util
import threading
from typing import Optional

# ---------------------------------------------------------------------
# 全局单例：语音引擎在模块首次导入时初始化一次，之后复用
# ---------------------------------------------------------------------

# 线程锁，防止多线程环境下并发调用 pyttsx3 导致异常
_lock = threading.Lock()

# 语音引擎实例（单例），初始化失败则为 None，表示降级到文本模式
_engine: Optional[object] = None

# pyttsx3 是否可用（是否安装了该库）
_pyttsx3_available = importlib.util.find_spec("pyttsx3") is not None


def _init_engine() -> bool:
    """初始化全局语音引擎（仅调用一次）。

    返回:
        bool: 初始化是否成功。失败时进入降级模式（只打印文本）。
    """
    global _engine
    if _engine is not None:
        # 已经初始化过，直接返回成功
        return True
    if not _pyttsx3_available:
        # 未安装 pyttsx3，降级
        return False
    try:
        import pyttsx3

        # 初始化 SAPI5 引擎（Windows）；macOS 用 'nsss'，Linux 用 'espeak'
        _engine = pyttsx3.init()
        # 默认语速 200 字/分钟（偏快，适合演示）
        _engine.setProperty("rate", 200)
        return True
    except Exception:
        # 初始化失败（如无声卡、无语音包等），降级
        _engine = None
        return False


# 模块加载时立即初始化一次引擎，确保全局单例
_engine_ok = _init_engine()


def is_available() -> bool:
    """返回当前离线 TTS 是否可用（库已安装且引擎初始化成功）。"""
    return _engine_ok and _engine is not None


# ---------------------------------------------------------------------
# 对外 API
# ---------------------------------------------------------------------

def set_rate(rate: int) -> None:
    """设置语音语速。

    参数:
        rate: 语速，单位为字/分钟。建议范围 100 ~ 300。
              默认值 200，数值越大越快。
    """
    if not is_available():
        return
    try:
        _engine.setProperty("rate", int(rate))  # type: ignore[union-attr]
    except Exception:
        # 设置失败不影响播报，忽略即可
        pass


def get_rate() -> int:
    """获取当前语速（字/分钟）。不可用时返回 0。"""
    if not is_available():
        return 0
    try:
        return int(_engine.getProperty("rate"))  # type: ignore[union-attr]
    except Exception:
        return 0


def speak(text: str) -> bool:
    """用离线语音引擎播报一段中文文本。

    全局单例引擎，加锁保证线程安全。
    若引擎不可用或播报失败，降级为打印文本，永不抛出异常。

    参数:
        text: 要播报的中文文本。空字符串则不做任何操作。

    返回:
        bool: 是否成功使用语音播报（True=语音已播放，False=降级为文本）。
    """
    text = str(text or "").strip()
    if not text:
        return False

    if is_available():
        try:
            with _lock:
                # 单例引擎复用，避免每次都 init/stop
                _engine.say(text)  # type: ignore[union-attr]
                _engine.runAndWait()  # type: ignore[union-attr]
            return True
        except Exception:
            # 播报失败（如设备被占用），降级到文本
            pass

    # 降级：打印到控制台
    try:
        print(f"[tts] {text}")
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------
# 测试代码：直接运行本文件即可听到测试语音
# ---------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("离线语音播报模块测试")
    print("=" * 50)

    if is_available():
        print(f"✓ pyttsx3 离线 TTS 可用，当前语速：{get_rate()} 字/分钟")
    else:
        print("✗ pyttsx3 不可用，已降级为文本模式")
        print("  请运行：pip install pyttsx3")

    print()

    # --- 测试 1：默认语速 ---
    print("[测试 1/3] 默认语速播报...")
    speak("物体识别系统启动成功")

    # --- 测试 2：慢速播报 ---
    print("[测试 2/3] 慢速播报（120字/分钟）...")
    set_rate(120)
    speak("这可能是一个苹果")

    # --- 测试 3：快速播报 ---
    print("[测试 3/3] 快速播报（250字/分钟）...")
    set_rate(250)
    speak("识别完成，相似度百分之八十七")

    # 恢复默认语速
    set_rate(200)

    print()
    print("测试完成！")
