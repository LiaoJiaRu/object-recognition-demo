# 现场演示清单（Demo Checklist）

> 目标：8~10 分钟内完整展示"录入新物体 → 正常识别 → 未知物体拒识"闭环，并能回答代码层面的提问。
> 建议全程使用命令行（最稳、出结果最快），Gradio / 摄像头作为加分展示。

## 0. 演示前准备（提前 30 分钟完成）

- [ ] 电脑插好电源，确认网络（仅首次加载 ResNet18 权重可能需要网络；权重已缓存则无需联网）
- [ ] 确认麦克风/摄像头权限（如要演示语音/摄像头；不演示可跳过）
- [ ] 手机/数据线准备好 3 类照片：1 个**新物体**、1 个**已注册物体**、1 个**未知物体**
- [ ] 确认 `config.yaml` 中 `data_root` 指向正确：
      `E:/cv_data/object_recognition_demo`
- [ ] 终端 `cd` 到项目根目录 `e:/projects/object_recognition_demo`

### 演示前自检（命令照抄，全部成功才开始）

```bash
python scripts/check_dataset.py     # 看到 [OK] 数据集检查通过
python scripts/evaluate.py          # registered_acc=1.0, unknown_recall=1.0
pytest -q                           # 81 passed
python app_cli.py list              # 确认特征库状态（空库或已有类别都可）
```

- [ ] 浏览器预打开一次 `python app.py` → http://127.0.0.1:7860 ，确认页面正常后关闭（Ctrl+C 停服务）
- [ ] 关闭无关程序降低 CPU 占用；准备一张白纸/纯色背景方便取景

---

## 1. 录入一个新物体

**讲述要点**：系统不是"训练新模型"，而是用预训练 ResNet18 把图片变成 512 维特征向量，存进特征库；同一物体可以存多张图。

### 方式 A：命令行（推荐，稳）

```bash
# 提前在 data_root 之外准备好新物体的 3~5 张照片，放到一个文件夹，例如：
# E:/cv_data/object_recognition_demo/gallery/demo_key/key_1.jpg ...
python app_cli.py enroll --name demo_key --dir E:/cv_data/object_recognition_demo/gallery/demo_key
python app_cli.py list
```

- [ ] 展示输出"新增 N 张""累计 N 条特征"
- [ ] `list` 中出现新类别
- [ ] 补充说明：特征已落盘到 `data_root/features/gallery.npz + labels.json`，重启程序仍在

### 方式 B：Gradio 界面（可视效果好）

```bash
python app.py
```

- [ ] 「录入」页上传一张新物体照片
- [ ] 名称框输入 `demo_key`（有麦克风可点麦克风 →「语音转名称」展示语音输入）
- [ ] 点击「录入」，下方列表实时出现该类别
- [ ] 回答预案：**"语音识别失败怎么办？"** → 自动降级为手动输入/文本提示，不影响录入

---

## 2. 正常识别

```bash
# 识别一张刚录入物体的新角度照片（不要和注册图完全相同，更有说服力）
python app_cli.py recognize --image <新物体的另一张照片路径> -v
```

- [ ] 输出 `label: demo_key`，`score` ≥ 0.75
- [ ] `-v` 展示 top-5 都是该类别 / 相似类别，解释"余弦相似度越高越像"
- [ ] 再识别一个其他已注册物体（如 apple），证明多类可用：
      `python app_cli.py recognize --image E:/cv_data/object_recognition_demo/test/apple/xxx.jpg -v`

Gradio：「识别」页上传图片 → 显示 label / score / top-k，并语音播报"识别结果，xxx"。

- [ ] 回答预案：**"score 是什么？"** → 查询图特征与库内特征的余弦相似度（-1~1，越接近 1 越相似）
- [ ] 回答预案：**"为什么不用分类 softmax？"** → 注册式系统类别开放，用检索+阈值可以随时加入新物体且无需重新训练

---

## 3. 未知物体判断（本项目核心亮点）

```bash
# test/unknown 下的图片均未在 gallery 注册
python app_cli.py recognize --image E:/cv_data/object_recognition_demo/test/unknown/3.jpg -v
```

- [ ] 输出 `label: unknown`，score 约 0.6x（低于阈值 0.75）
- [ ] 强调：**即使最像的库类别也低于阈值，系统选择拒识而不是乱猜**

### 阈值互动（如果时间允许，强烈推荐）

- [ ] 打开 `config.yaml`，指出 `threshold: 0.75` 这一行
- [ ] 讲解：已注册类最低分 0.77，unknown 最高分 0.68，阈值卡在中间
- [ ] 可选：临时把 threshold 改成 0.5 再识别 unknown 图 → 会被误识别；改回 0.75 → 恢复 unknown
      （改完务必改回 0.75！）

### 自动评估（用数据说话）

```bash
python scripts/evaluate.py
```

- [ ] 展示终端输出：registered_acc = 1.0000（8/8），unknown_recall = 1.0000（4/4）
- [ ] 打开 `data_root/reports/eval_report.json`，指出每张图的 path/truth/pred/score/ok
- [ ] 强调：**真值自动来自文件夹名，不需要手工标注，也不依赖 data_index.json**

### 摄像头实时识别（加分项，机器有摄像头再演示）

```bash
python webcam_app.py        # 先确保已录入类别；每 5 帧识别一次，按 q 退出
```

- [ ] 对准已注册物体：左上角绿色 label + score
- [ ] 移出画面 / 对准手或脸：红色 unknown
- [ ] 按 q 退出，讲解"每 5 帧推理一次是为了降低 CPU 压力，其余帧沿用上次结果"

---

## 4. 代码解释准备（评委可能提问）

按"一条数据流"准备，能在 1 分钟内讲清：

1. **图片怎么变成特征？**
   `core/feature_extractor.py`：ResNet18 + `model.fc = nn.Identity()` → 512 维 → L2 归一化；
   预处理 Resize(256)/CenterCrop(224)/ToTensor/Normalize(ImageNet)。

2. **识别怎么算？**
   `core/recognizer.py`：库特征矩阵 × 查询向量 = 余弦相似度数组 → top-k →
   最高分 ≥ threshold 输出类别，否则 unknown；空库也输出 unknown。

3. **特征存哪、重启怎么办？**
   `core/gallery.py`：`gallery.npz`（特征矩阵）+ `labels.json`（标签和路径）成对落盘，
   启动时 load，条数一致性校验。

4. **数据路径写死了吗？**
   没有。全部从 `config.yaml` 的 `data_root` 拼出；改一个字段即可迁移数据目录。
   CLI 还支持 `--data-root` 临时覆盖。

5. **语音/摄像头坏了影响识别吗？**
   不影响。`core/voice.py` 全程 try/except 降级（键盘/打印）；
   摄像头单帧推理失败也只告警。自动测试全部用 mock，不依赖麦克风和摄像头。

6. **怎么保证代码是对的？**
   `pytest -q` 共 81 个用例：阈值边界、空库、持久化重启、真值递归规则、
   降级路径等；tmp_path 合成数据不污染真实数据。

7. **高频追问速答**：
   - "为什么 L2 归一化？" → 归一化后点积就是余弦相似度，只比方向不比模长。
   - "top-k 有什么用？" → 给出候选列表便于人工复核，最终类别取 top1。
   - "阈值怎么定的？" → 当前数据上已注册类最低分 0.77、unknown 最高 0.68，取 0.75；
     数据更多时应在验证集上按错误率权衡自动选取。
   - "CPU 能跑吗？" → 能，模型 11M 参数，单张推理亚秒级；摄像头模式每 5 帧推一次进一步减负。

---

## 5. 演示后收尾

- [ ] 若演示中录入了临时类别，删除并确认：
      `python app_cli.py delete --name demo_key` → `python app_cli.py list`
- [ ] 若改过 `config.yaml` 的 threshold，确认恢复为 `0.75`
- [ ] 关闭 app.py / webcam_app.py（Ctrl+C 或 q）
- [ ] 重新跑一次 `pytest -q`，确认环境干净（可选）

## 6. 应急预案

| 突发情况 | 处理 |
|---|---|
| Gradio 起不来/浏览器打不开 | 立即改用命令行四件套，功能完全等价 |
| 麦克风无反应 | 跳过语音，手动打字；一句话说明"语音是可选旁路，已自动降级" |
| 摄像头打不开 | 跳过实时环节，用 `recognize` 命令展示同样的识别逻辑 |
| 识别结果不对 | 先 `list` 确认类别已录入；展示 `-v` top-k；强调阈值与拍摄角度，必要时换注册时同角度照片 |
| 现场没有网络 | ResNet18 权重本地已缓存，STT 在线语音跳过，其余全离线可用 |
