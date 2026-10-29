"""tts_demo.py

浏览器端语音播报最小演示（Web Speech API）。
不依赖 torch / pyttsx3，纯浏览器端语音合成，完全离线。
用来验证 app.py 中语音播报模块的效果。

启动:
    python tts_demo.py
"""
from __future__ import annotations

import gradio as gr


def make_demo() -> gr.Blocks:
    with gr.Blocks(title="语音播报测试") as demo:
        gr.Markdown("# 🔊 浏览器端语音播报测试")
        gr.Markdown(
            "使用浏览器自带 **Web Speech API**，"
            "纯离线，无需联网，在当前设备发声。"
        )

        with gr.Row():
            with gr.Column():
                test_text = gr.Textbox(
                    label="要播报的文本",
                    value="识别结果，这可能是苹果，相似度百分之八十七",
                    lines=3,
                )
                with gr.Row():
                    tts_rate = gr.Slider(
                        minimum=0.5, maximum=2.0, value=1.0, step=0.1,
                        label="语速", info="1.0 为正常速度",
                    )
                play_btn = gr.Button("🔊 播放", variant="primary", size="lg")

            with gr.Column():
                gr.Markdown("### 测试场景")
                with gr.Row():
                    test1_btn = gr.Button("场景1：识别成功（苹果）")
                    test2_btn = gr.Button("场景2：识别失败（unknown）")
                with gr.Row():
                    test3_btn = gr.Button("场景3：慢速播报")
                    test4_btn = gr.Button("场景4：快速播报")
                result_display = gr.Markdown("点击上方按钮测试")

        # 隐藏文本框：实际播报的内容
        speak_text = gr.Textbox(visible=False, label="播报文本")

        # --- 浏览器端语音播报 JS ---
        _speak_js = """
        (text, rate) => {
            if (!text) return;
            if (!('speechSynthesis' in window)) {
                alert('当前浏览器不支持语音合成，请使用 Chrome / Edge / Safari 等现代浏览器');
                return;
            }
            // 取消之前的播放，防止堆叠
            window.speechSynthesis.cancel();
            const utter = new SpeechSynthesisUtterance(text);
            utter.lang = 'zh-CN';
            utter.rate = parseFloat(rate) || 1.0;
            utter.pitch = 1.0;
            utter.volume = 1.0;
            window.speechSynthesis.speak(utter);
        }
        """

        # 手动播放按钮
        play_btn.click(
            fn=None,
            inputs=[test_text, tts_rate],
            outputs=[],
            js=_speak_js,
        )

        # 场景测试按钮
        def _set_scene(text: str, rate: float, desc: str) -> tuple[str, float, str]:
            return text, rate, desc

        test1_btn.click(
            fn=lambda: ("识别结果，这可能是苹果，相似度百分之八十七", 1.0,
                        "✅ 场景1：正常语速播报识别结果"),
            outputs=[speak_text, tts_rate, result_display],
        )
        test2_btn.click(
            fn=lambda: ("未能识别出已知物体", 1.0,
                        "✅ 场景2：播报 unknown 结果"),
            outputs=[speak_text, tts_rate, result_display],
        )
        test3_btn.click(
            fn=lambda: ("识别结果，这可能是一个杯子", 0.6,
                        "✅ 场景3：慢速播报（0.6倍速）"),
            outputs=[speak_text, tts_rate, result_display],
        )
        test4_btn.click(
            fn=lambda: ("识别结果，这可能是一本书，相似度百分之九十二", 1.8,
                        "✅ 场景4：快速播报（1.8倍速）"),
            outputs=[speak_text, tts_rate, result_display],
        )

        # 场景文本变化时自动播报
        speak_text.change(
            fn=None,
            inputs=[speak_text, tts_rate],
            outputs=[],
            js=_speak_js,
        )

    return demo


if __name__ == "__main__":
    demo = make_demo()
    demo.launch(server_name="127.0.0.1", server_port=7861)
