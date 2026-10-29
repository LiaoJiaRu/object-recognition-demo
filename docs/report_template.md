# 计算机视觉课程项目报告

**项目名称：基于 ResNet18 和语音交互的物体识别 Demo**

| 项目 | 内容 |
|---|---|
| 学院 / 专业 / 班级 | （填写） |
| 小组成员 | （填写） |
| 指导教师 | （填写） |
| 完成日期 | 2026 年 __ 月 __ 日 |

> 本模板中带「（实测）」的数字为本仓库当前数据上的真实运行结果，可直接引用；带「（填写）」的部分请小组补充。

---

## 一、系统功能

本系统是一个 CPU 可运行的物体识别 Demo，实现了"注册 → 识别 → 未知拒识 → 持久化 → 交互"的完整闭环：

1. **特征提取**：使用 ImageNet 预训练 ResNet18，去掉最后全连接层，输出 512 维特征并做 L2 归一化。
2. **物体录入（enroll）**：为每个类别提供若干注册图，逐张提取特征存入特征库；同一类别支持多张图、多个特征，可重复录入累加。
3. **物体识别（recognize）**：对查询图片计算与库内全部特征的余弦相似度，返回 top-k 检索结果及最高分类别。
4. **未知物体判断**：最高相似度低于阈值（`config.yaml: threshold = 0.75`）时输出 `unknown`；特征库为空时一律输出 `unknown`。
5. **特征库持久化**：特征保存为 `<data_root>/features/gallery.npz`，标签与图片路径保存为 `labels.json`，程序重启后自动恢复。
6. **自动评估**：遍历测试集，按"文件夹名即真值"的规则自动比对，生成 `reports/eval_report.json`。
7. **多形态交互**：
   - 命令行（`app_cli.py`：enroll / recognize / list / delete）；
   - Gradio Web 界面（`app.py`：图片上传、列表管理、删除）；
   - 语音交互：录入时可用麦克风说出类别名，识别后 TTS 播报，语音不可用时自动降级为键盘/文本，不影响核心功能；
   - 摄像头实时识别（`webcam_app.py`）：每 5 帧识别一次以降低 CPU 压力，画面叠加 label / score，按 q 退出。
8. **配置与数据分离**：数据目录独立于代码目录，所有路径、阈值、top-k 均从 `config.yaml` 读取，代码中无硬编码数据路径。

## 二、实现方法

### 2.1 总体架构

系统按单一职责分为 5 个核心模块 + 3 个入口：

| 文件 | 职责 |
|---|---|
| `core/feature_extractor.py` | `FeatureExtractor`：ResNet18 去 FC、ImageNet 预处理、512 维 L2 归一化特征、单张/批量提取 |
| `core/gallery.py` | `FeatureGallery`：add / delete / list / get_all / save / load，npz + json 成对持久化 |
| `core/recognizer.py` | `Recognizer`：余弦相似度 top-k 检索、阈值 unknown 判定、空库判定 |
| `core/voice.py` | `VoiceIO`：STT / TTS 门面，懒加载可选依赖，全程 try/except 降级 |
| `core/webcam.py` | 帧节拍器 `InferenceGate`、BGR 帧→特征、画面文字叠加、按键判断（均可单测） |
| `app_cli.py` / `app.py` / `webcam_app.py` | 命令行 / Gradio / 摄像头三种入口，均只组合调用 core 模块 |
| `scripts/check_dataset.py` / `scripts/evaluate.py` | 数据组织校验 / 自动评估 |

### 2.2 特征提取

- 模型：`torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)`，`model.fc = nn.Identity()`，`eval()` 模式，推理包裹 `torch.no_grad()`。
- 预处理：`Resize(256) → CenterCrop(224) → ToTensor → Normalize(ImageNet 均值方差)`。
- 输出：512 维向量，按行做 L2 归一化（除以 L2 范数），使后续点积等价于余弦相似度。
- 计算设备默认 CPU，低算力优先。

### 2.3 识别与 unknown 判定

对查询特征 **q** 与库内特征矩阵 **F**（均已 L2 归一化）：

```
相似度 s = F · q          # numpy 矩阵-向量乘，一次得到与全部注册特征的余弦相似度
topk  = argsort(-s)[:k]   # k 来自 config.yaml（默认 5）
若 max(s) >= threshold    → 输出 top1 类别名
否则                       → 输出 "unknown"
特征库为空                 → 输出 "unknown"，score = 0
```

阈值法开放集识别简单高效：阈值以上认为是已注册物体，阈值以下拒识为未知物体。

### 2.4 特征库持久化

- `gallery.npz`：用 `np.savez` 保存形状 (N, 512) 的特征矩阵；
- `labels.json`：保存与特征行对齐的标签列表、图片路径列表；
- 加载时校验三者条数一致，避免标签错位；
- 录入采用"先 load 再 append 再 save"，支持跨进程重启与同类多图累加。

### 2.5 真值规则与评估指标

- 真值**永远来自文件夹名**（`data_index.json` 仅为统计记录，不参与评估）：
  - 测试图片相对 `test/` 的路径中任一段为 `unknown`（含其嵌套子文件夹）→ 真值 `unknown`；
  - 否则取直接父文件夹名。
- 指标：
  - `registered_acc` = 已注册类测试图预测正确数 / 已注册类测试图总数；
  - `unknown_recall` = unknown 图被正确拒识数 / unknown 图总数；
  - `unknown_false_accept` = unknown 图被误接受数 / unknown 图总数（= 1 − unknown_recall）；
  - `details`：逐图记录 path / truth / pred / score / ok。

### 2.6 语音与摄像头的降级设计

- `VoiceIO` 对 speech_recognition / pyttsx3 / pyaudio 采用**懒加载**：已安装则真实语音，未安装或调用失败则分别退化为键盘输入与文本打印；所有异常被捕获，**语音失败不会影响识别结果**。
- 摄像头采用**单线程同步采集**（避免多线程竞争 VideoCapture 导致黑帧），由 `InferenceGate` 保证第 1 帧立即识别、之后每 5 帧一次，非识别帧沿用上一次结果；单帧推理异常只告警不中断。

## 三、测试结果

### 3.1 数据集（实测）

- gallery：5 类（apple / banana / bottle / cup / white radish），共 38 张注册图；
- test：5 个已注册类别共 8 张测试图；test/unknown 共 4 张未知图。

### 3.2 自动评估（实测，threshold=0.75，topk=5）

| 指标 | 结果 |
|---|---|
| registered_acc（已注册类准确率） | **1.0000（8/8）** |
| unknown_recall（未知拒识率） | **1.0000（4/4）** |
| unknown_false_accept（未知误接受率） | **0.0000（0/4）** |

分数间隔（实测）：已注册类最低分 0.7707，unknown 最高分 0.6753，阈值 0.75 位于两者之间，留有约 0.10 的区分裕度。

### 3.3 自动化测试（实测）

```
pytest -q  →  81 passed
```

| 测试文件 | 用例数 | 覆盖内容 |
|---|---:|---|
| test_check_dataset.py | 10 | 数据阈值、后缀过滤、unknown 递归统计 |
| test_feature_extractor.py | 4 | 512 维形状、L2 归一化、同图/异图相似度 |
| test_gallery.py | 9 | 多图累加、save/load 重启恢复、删除、空库 |
| test_recognizer.py | 10 | 余弦 top-k、阈值边界、空库、真实数据联调 |
| test_evaluate_accuracy.py | 7 | 真值规则、指标数学、真实数据准确率门槛 |
| test_cli.py | 12 | 命令行闭环、`--data-root` 覆盖、错误退出码 |
| test_voice.py | 16 | STT/TTS 全部降级路径（mock，无麦克风依赖） |
| test_webcam_utils.py | 13 | 帧节拍、帧特征、叠加颜色、q 键（不开摄像头） |

测试原则：除特征提取/识别用真实模型与真实数据做联调外，其余一律用 `tmp_path` 合成数据与 mock 后端，不污染真实数据、不依赖麦克风/摄像头/网络。

### 3.4 复现实验命令

```bash
python scripts/check_dataset.py
python scripts/evaluate.py
pytest -q
python app.py            # Gradio：http://127.0.0.1:7860
python webcam_app.py     # 摄像头（有摄像头的机器）
```

### 3.5 不足与改进方向（填写/参考）

- 注册图与测试图规模较小，指标的统计意义有限；
- 阈值 0.75 为经验值，可在更大数据集上用 ROC / 验证集自动选取；
- 目前类别表征为"每图一特征"的最近邻检索，可改进为类原型（类内特征均值）以加速检索；
- 可加入数据增强、度量学习（如 ArcFace）提升类间分离度。

## 四、成员分工

| 成员 | 学号 | 主要分工 | 完成模块/文档 |
|---|---|---|---|---:|
| （填写） | （填写） | 特征提取与模型 | core/feature_extractor.py 等 |
| （填写） | （填写） | 特征库与识别器 | core/gallery.py、core/recognizer.py |
| （填写） | （填写） | 评估与测试 | scripts/evaluate.py、tests/ |
| （填写） | （填写） | 界面、语音与摄像头 | app.py、core/voice.py、webcam_app.py |
| （填写） | （填写） | 文档与演示 | README、本报告、demo_checklist |

## 五、参考资料

1. He K. et al. *Deep Residual Learning for Image Recognition*. CVPR 2016.
2. PyTorch / torchvision 官方文档：`torchvision.models.resnet18`。
3. Gradio 官方文档：Blocks、Image/Audio 组件。
4. OpenCV-Python 官方教程：VideoCapture、putText。
