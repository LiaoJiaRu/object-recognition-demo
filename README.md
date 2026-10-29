# 基于 ResNet18 和语音交互的物体识别 Demo

计算机视觉课程项目：使用 ImageNet 预训练 ResNet18 提取 512 维特征并做 L2 归一化，基于余弦相似度 top-k 检索实现物体识别，低于阈值判为 unknown；特征库持久化，提供命令行、Gradio 界面、可降级语音交互与摄像头实时识别。全程 CPU 可运行。

## 1. 环境安装

建议 Python 3.10+（开发机实测 3.12.7，Windows）。

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
# source .venv/bin/activate

pip install -r requirements.txt
```

主要依赖：`torch` / `torchvision`（CPU 版即可）、`numpy`、`pillow`、`pyyaml`、`gradio`、`opencv-python`、`pytest`。

开发机实测可用版本：

| 包 | 版本 |
|---|---|
| Python | 3.12.7 |
| torch | 2.8.0+cpu |
| torchvision | 0.23.0+cpu（须与 torch 主版本匹配） |
| numpy | 1.26.4 |
| gradio | 6.28.0 |
| opencv-python | 4.10.0.84 |

> 注意：torch 与 torchvision 版本必须配套（如 torch 2.8.x 配 torchvision 0.23.x），否则可能出现 `operator torchvision::nms does not exist` 等导入错误。
>
> 语音依赖（`SpeechRecognition`、`pyaudio`、`pyttsx3`）为**可选**，见 `requirements.txt` 末尾注释；未安装时自动降级为键盘输入/文本打印，不影响识别与自动测试。

## 2. 数据准备

数据目录与项目目录分离，**所有数据路径只从 `config.yaml` 读取，代码中不硬编码**。

```
<data_root>/
├── gallery/<类别名>/*.jpg     # 注册图，每类建议 >=5 张
├── test/<类别名>/*.jpg        # 已知类测试图（真值 = 父文件夹名）
├── test/unknown/**/*.jpg      # 未知测试图（含子文件夹，一律视为 unknown）
├── features/gallery.npz       # 持久化特征库（自动生成）
├── features/labels.json       # 标签与图片路径（自动生成）
└── reports/
    ├── data_index.json        # 数据统计记录（仅记录，不是真值来源）
    └── eval_report.json       # 评估报告（自动生成）
```

图片支持 `.jpg / .jpeg / .png`（大写后缀亦可），文件名可随机。真值规则永远是：路径中包含 `unknown` 文件夹 → `unknown`，否则取直接父文件夹名。

校验数据组织（类别数、每类图片数、unknown 图片数）：

```bash
python scripts/check_dataset.py
```

通过返回退出码 0，并打印统计信息与 `reports/data_index.json` 路径。

## 3. config.yaml 修改方法

`config.yaml` 是唯一配置入口。迁移数据目录时只改 `data_root` 一项即可：

```yaml
data_root: E:/cv_data/object_recognition_demo   # 改成你的数据根目录（正斜杠）
gallery_dir: gallery      # 相对 data_root
test_dir: test
feature_dir: features
report_dir: reports

image_size: 224
device: cpu               # 低算力优先

threshold: 0.75           # 余弦相似度低于此值输出 unknown
topk: 5                   # top-k 检索
unknown_label: unknown

feature_file: gallery.npz
labels_file: labels.json
report_file: eval_report.json
```

命令行工具还支持运行时临时覆盖而不改文件：

```bash
python app_cli.py --config config.yaml --data-root E:/other/data list
```

## 4. 运行命令

### 4.1 命令行（核心闭环）

```bash
# 录入一个类别（目录下所有图片提取特征并持久化，可重复录入同名类别累加）
python app_cli.py enroll --name apple --dir E:/cv_data/object_recognition_demo/gallery/apple

# 查看已注册类别及特征条数
python app_cli.py list

# 识别单张图片（-v 显示 top-k 明细）
python app_cli.py recognize --image E:/cv_data/object_recognition_demo/test/apple/xxx.jpg -v

# 删除一个类别
python app_cli.py delete --name apple
```

特征库保存在 `<data_root>/features/`，重启程序后自动加载。

### 4.2 Gradio 图形界面

```bash
python app.py
```

浏览器打开 http://127.0.0.1:7860 ，三个标签页：

- **录入**：上传图片 + 输入（或麦克风语音说出）类别名称 → 录入
- **识别**：上传图片 → 显示 label / score / top-k，并语音播报结果（无语音后端时文本降级）
- **管理**：查看已注册列表、删除类别

### 4.3 摄像头实时识别（加分项）

```bash
python webcam_app.py                 # 默认摄像头 0，每 5 帧识别一次，按 q 退出
python webcam_app.py --camera 0 --every 5
```

画面左上角实时显示 label / score（已知类绿色、unknown 红色）。使用前请先录入至少一个类别。

### 4.4 自动评估

```bash
python scripts/evaluate.py
```

遍历 `test/` 下全部图片，按文件夹真值统计并输出 `reports/eval_report.json`（含 `registered_acc`、`unknown_recall`、`unknown_false_accept` 和逐图 details）。

### 4.5 视频测试集评估

```bash
python scripts/evaluate_videos.py
```

遍历 `<data_root>/<video_test_dir>/<类别名>/*.mp4`（也支持 avi/mov/mkv/webm），每个视频按 `frame_interval` 抽帧逐帧识别并多数投票，按文件夹真值统计视频级 top-1 准确率，输出 `reports/video_eval_report.json`（`max_frames` 限制单视频抽帧上限，0 为不限制）。视频损坏或 0 帧会记为 read_error 并计入错误，不中断整体评估。

## 5. 测试命令

```bash
# 全部自动测试（不依赖麦克风、摄像头、网络）
pytest -q

# 按模块运行
pytest tests/test_check_dataset.py -q       # 数据检查
pytest tests/test_feature_extractor.py -q   # 特征提取（真实 ResNet18）
pytest tests/test_gallery.py -q             # 特征库持久化（tmp_path）
pytest tests/test_recognizer.py -q          # 识别与 unknown 判定
pytest tests/test_evaluate_accuracy.py -q   # 评估指标与真值规则
pytest tests/test_cli.py -q                 # 命令行闭环
pytest tests/test_voice.py -q               # 语音降级逻辑（mock）
pytest tests/test_webcam_utils.py -q        # 摄像头工具函数（不开摄像头）
pytest tests/test_video.py -q               # 视频抽帧/多数投票逻辑
pytest tests/test_evaluate_videos.py -q     # 视频测试集评估（合成视频+假帧函数）
```

## 6. 项目结构

```
object_recognition_demo/
├── config.yaml                  # 唯一配置入口（路径/阈值/topk）
├── requirements.txt
├── app.py                       # Gradio 界面（录入/识别/管理 + 语音）
├── app_cli.py                   # 命令行：enroll / recognize / list / delete
├── webcam_app.py                # OpenCV 摄像头实时识别（按 q 退出）
├── tts_demo.py                  # 浏览器 Web Speech API 语音播报测试页（独立运行）
├── utils/
│   └── tts.py                   # pyttsx3 离线 TTS 模块（独立自测/备用；Web 界面播报走浏览器合成）
├── core/
│   ├── feature_extractor.py     # ResNet18 去 FC，512 维 L2 归一化特征
│   ├── gallery.py               # FeatureGallery：增删存取 + npz/json 持久化
│   ├── recognizer.py            # Recognizer：余弦相似度 top-k + 阈值 unknown
│   ├── voice.py                 # VoiceIO：STT/TTS，失败自动降级
│   ├── video.py                 # 视频抽帧与多数投票聚合（纯逻辑）
│   └── webcam.py                # 帧节拍/帧识别/画面叠加等可测试工具
├── scripts/
│   ├── check_dataset.py         # 数据组织校验
│   ├── evaluate.py              # 自动评估并生成 eval_report.json
│   └── evaluate_videos.py       # 视频测试集评估，生成 video_eval_report.json
├── tests/                       # 10 个测试文件，tmp_path 合成数据为主
└── docs/
    ├── report_template.md       # 课程报告模板
    └── demo_checklist.md        # 现场演示清单
```

## 7. 识别原理简述

1. ResNet18（ImageNet 预训练）去掉最后全连接层，对 224×224 输入输出 512 维特征并 L2 归一化；
2. 录入时把每张注册图的特征连同类别名、图片路径存入特征库；
3. 识别时计算查询特征与库内全部特征的点积（即余弦相似度），取 top-k；
4. 最高相似度 ≥ `threshold` → 输出对应类别，否则输出 `unknown`；特征库为空也判 `unknown`。
