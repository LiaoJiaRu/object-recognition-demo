"""tests/test_tts.py

离线 TTS 模块单元测试。
全部通过 monkeypatch 注入假引擎验证逻辑，不依赖声卡 / 语音包。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import utils.tts as tts_mod  # noqa: E402


# ---------------------------------------------------------------------
# 假引擎：模拟 pyttsx3 engine 对象
# ---------------------------------------------------------------------
class FakeEngine:
    def __init__(self) -> None:
        self.rate = 200
        self.spoken: list[str] = []
        self.say_calls = 0
        self.run_calls = 0
        self.raise_on_say = False
        self.raise_on_run = False

    def say(self, text: str) -> None:
        if self.raise_on_say:
            raise RuntimeError("say failed")
        self.say_calls += 1
        self.spoken.append(text)

    def runAndWait(self) -> None:  # noqa: N802 (pyttsx3 原生命名)
        if self.raise_on_run:
            raise RuntimeError("runAndWait failed")
        self.run_calls += 1

    def setProperty(self, name: str, value) -> None:  # noqa: N802
        if name == "rate":
            self.rate = int(value)

    def getProperty(self, name: str):  # noqa: N802
        if name == "rate":
            return self.rate
        return None


# ---------------------------------------------------------------------
# 测试辅助：把全局引擎替换成假的
# ---------------------------------------------------------------------
@pytest.fixture
def fake_engine(monkeypatch):
    """注入假引擎到全局单例，使 speak 走真实播报分支（但用假后端）。"""
    fake = FakeEngine()
    monkeypatch.setattr(tts_mod, "_engine", fake)
    monkeypatch.setattr(tts_mod, "_engine_ok", True)
    return fake


# ---------------------------------------------------------------------
# speak 基本功能
# ---------------------------------------------------------------------
def test_speak_normal_text_uses_engine(fake_engine):
    """正常文本应调用引擎播报。"""
    result = tts_mod.speak("你好世界")
    assert result is True
    assert fake_engine.spoken == ["你好世界"]
    assert fake_engine.say_calls == 1
    assert fake_engine.run_calls == 1


def test_speak_empty_text_is_noop(fake_engine):
    """空文本不调用引擎，也不打印。"""
    result = tts_mod.speak("")
    assert result is False
    assert fake_engine.say_calls == 0
    assert fake_engine.run_calls == 0


def test_speak_whitespace_text_is_noop(fake_engine):
    """纯空白文本视为空。"""
    result = tts_mod.speak("   \n\t  ")
    assert result is False
    assert fake_engine.say_calls == 0


def test_speak_none_text_is_noop(fake_engine):
    """None 视为空文本。"""
    result = tts_mod.speak(None)  # type: ignore[arg-type]
    assert result is False
    assert fake_engine.say_calls == 0


# ---------------------------------------------------------------------
# 降级逻辑
# ---------------------------------------------------------------------
def test_speak_falls_back_to_print_when_engine_unavailable(capsys, monkeypatch):
    """引擎不可用时降级为控制台打印。"""
    monkeypatch.setattr(tts_mod, "_engine", None)
    monkeypatch.setattr(tts_mod, "_engine_ok", False)

    result = tts_mod.speak("测试降级")
    assert result is False
    captured = capsys.readouterr()
    assert "[tts] 测试降级" in captured.out


def test_speak_falls_back_on_engine_error(capsys, fake_engine):
    """引擎播报出错时降级为打印，不抛异常。"""
    fake_engine.raise_on_run = True

    result = tts_mod.speak("出错了")
    assert result is False
    captured = capsys.readouterr()
    assert "[tts] 出错了" in captured.out


def test_speak_never_raises_even_print_broken(monkeypatch, fake_engine):
    """极端情况：引擎坏了 + print 也坏了，仍不抛异常。"""
    fake_engine.raise_on_say = True

    def boom_print(_text):
        raise RuntimeError("stdout broken")

    monkeypatch.setattr("builtins.print", boom_print)

    # 不抛异常即通过
    result = tts_mod.speak("极端测试")
    assert result is False


# ---------------------------------------------------------------------
# 语速调节
# ---------------------------------------------------------------------
def test_set_rate_changes_engine_rate(fake_engine):
    """set_rate 应修改引擎语速。"""
    tts_mod.set_rate(150)
    assert fake_engine.rate == 150


def test_get_rate_returns_current_rate(fake_engine):
    """get_rate 应返回当前语速。"""
    fake_engine.rate = 180
    assert tts_mod.get_rate() == 180


def test_set_rate_when_unavailable_noop(monkeypatch):
    """引擎不可用时 set_rate 不报错、不做事。"""
    monkeypatch.setattr(tts_mod, "_engine", None)
    monkeypatch.setattr(tts_mod, "_engine_ok", False)
    # 不抛异常即通过
    tts_mod.set_rate(200)


def test_get_rate_when_unavailable_returns_zero(monkeypatch):
    """引擎不可用时 get_rate 返回 0。"""
    monkeypatch.setattr(tts_mod, "_engine", None)
    monkeypatch.setattr(tts_mod, "_engine_ok", False)
    assert tts_mod.get_rate() == 0


# ---------------------------------------------------------------------
# 可用性检测
# ---------------------------------------------------------------------
def test_is_available_true_when_engine_ok(fake_engine):
    assert tts_mod.is_available() is True


def test_is_available_false_when_no_engine(monkeypatch):
    monkeypatch.setattr(tts_mod, "_engine", None)
    monkeypatch.setattr(tts_mod, "_engine_ok", False)
    assert tts_mod.is_available() is False


# ---------------------------------------------------------------------
# 线程安全（简单验证：锁存在且可用）
# ---------------------------------------------------------------------
def test_lock_exists_and_is_reentrant_safe(fake_engine):
    """多次并发调用（串行模拟）不会导致状态混乱。"""
    texts = ["第一句", "第二句", "第三句"]
    for t in texts:
        tts_mod.speak(t)
    assert fake_engine.spoken == texts
    assert fake_engine.say_calls == 3
    assert fake_engine.run_calls == 3
