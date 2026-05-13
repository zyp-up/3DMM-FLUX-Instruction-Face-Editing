<p align="center">
  <h1 align="center">ControlFace × FLUX.2: Text-Driven Facial Expression Editing</h1>
  <h3 align="center">基于 DECA 几何控制 + FLUX.2-klein 流匹配的人脸表情编辑</h3>
  <p align="center">本仓库基于 <a href="https://github.com/cvlab-kaist/ControlFace">ControlFace (CVPR 2025)</a> 的思想，将其控制范式迁移到 <b>FLUX.2-klein-base</b> 之上，并使用自建的 <b>FacePairEmoji</b> 表情配对数据集做端到端训练。</p>
</p>

<!-- demo-gallery-start -->
## 样例结果

下表的行和列使用同一表情顺序：`Neutral → Angry → Disgust → Fear → Happy → Sad → Surprise`。每一行对应同一个输入身份，左侧仅标注该样例的原始分辨率；浅黄色对角线单元格是原始输入图，其他单元格为四图对比：上排是本方法 `Control-CN / Control-EN`，下排是原始 FLUX.2 基线 `OG-CN / OG-EN`。

<table>
<tr><th>Resolution</th><th>Neutral</th><th>Angry</th><th>Disgust</th><th>Fear</th><th>Happy</th><th>Sad</th><th>Surprise</th></tr>
<tr><td align="center"><b>2048x2048</b></td><td align="center" bgcolor="#fff7df"><img src="docs/demo_grid/neutral_input.jpg" alt="Neutral input"></td><td align="center"><img src="docs/demo_grid/neutral_to_angry.jpg" alt="Neutral to Angry"></td><td align="center"><img src="docs/demo_grid/neutral_to_disgust.jpg" alt="Neutral to Disgust"></td><td align="center"><img src="docs/demo_grid/neutral_to_fear.jpg" alt="Neutral to Fear"></td><td align="center"><img src="docs/demo_grid/neutral_to_happy.jpg" alt="Neutral to Happy"></td><td align="center"><img src="docs/demo_grid/neutral_to_sad.jpg" alt="Neutral to Sad"></td><td align="center"><img src="docs/demo_grid/neutral_to_surprise.jpg" alt="Neutral to Surprise"></td></tr>
<tr><td align="center"><b>1328x1776</b></td><td align="center"><img src="docs/demo_grid/angry_to_neutral.jpg" alt="Angry to Neutral"></td><td align="center" bgcolor="#fff7df"><img src="docs/demo_grid/angry_input.jpg" alt="Angry input"></td><td align="center"><img src="docs/demo_grid/angry_to_disgust.jpg" alt="Angry to Disgust"></td><td align="center"><img src="docs/demo_grid/angry_to_fear.jpg" alt="Angry to Fear"></td><td align="center"><img src="docs/demo_grid/angry_to_happy.jpg" alt="Angry to Happy"></td><td align="center"><img src="docs/demo_grid/angry_to_sad.jpg" alt="Angry to Sad"></td><td align="center"><img src="docs/demo_grid/angry_to_surprise.jpg" alt="Angry to Surprise"></td></tr>
<tr><td align="center"><b>1184x1392</b></td><td align="center"><img src="docs/demo_grid/disgust_to_neutral.jpg" alt="Disgust to Neutral"></td><td align="center"><img src="docs/demo_grid/disgust_to_angry.jpg" alt="Disgust to Angry"></td><td align="center" bgcolor="#fff7df"><img src="docs/demo_grid/disgust_input.jpg" alt="Disgust input"></td><td align="center"><img src="docs/demo_grid/disgust_to_fear.jpg" alt="Disgust to Fear"></td><td align="center"><img src="docs/demo_grid/disgust_to_happy.jpg" alt="Disgust to Happy"></td><td align="center"><img src="docs/demo_grid/disgust_to_sad.jpg" alt="Disgust to Sad"></td><td align="center"><img src="docs/demo_grid/disgust_to_surprise.jpg" alt="Disgust to Surprise"></td></tr>
<tr><td align="center"><b>832x1248</b></td><td align="center"><img src="docs/demo_grid/fear_to_neutral.jpg" alt="Fear to Neutral"></td><td align="center"><img src="docs/demo_grid/fear_to_angry.jpg" alt="Fear to Angry"></td><td align="center"><img src="docs/demo_grid/fear_to_disgust.jpg" alt="Fear to Disgust"></td><td align="center" bgcolor="#fff7df"><img src="docs/demo_grid/fear_input.jpg" alt="Fear input"></td><td align="center"><img src="docs/demo_grid/fear_to_happy.jpg" alt="Fear to Happy"></td><td align="center"><img src="docs/demo_grid/fear_to_sad.jpg" alt="Fear to Sad"></td><td align="center"><img src="docs/demo_grid/fear_to_surprise.jpg" alt="Fear to Surprise"></td></tr>
<tr><td align="center"><b>656x896</b></td><td align="center"><img src="docs/demo_grid/happy_to_neutral.jpg" alt="Happy to Neutral"></td><td align="center"><img src="docs/demo_grid/happy_to_angry.jpg" alt="Happy to Angry"></td><td align="center"><img src="docs/demo_grid/happy_to_disgust.jpg" alt="Happy to Disgust"></td><td align="center"><img src="docs/demo_grid/happy_to_fear.jpg" alt="Happy to Fear"></td><td align="center" bgcolor="#fff7df"><img src="docs/demo_grid/happy_input.jpg" alt="Happy input"></td><td align="center"><img src="docs/demo_grid/happy_to_sad.jpg" alt="Happy to Sad"></td><td align="center"><img src="docs/demo_grid/happy_to_surprise.jpg" alt="Happy to Surprise"></td></tr>
<tr><td align="center"><b>448x592</b></td><td align="center"><img src="docs/demo_grid/sad_to_neutral.jpg" alt="Sad to Neutral"></td><td align="center"><img src="docs/demo_grid/sad_to_angry.jpg" alt="Sad to Angry"></td><td align="center"><img src="docs/demo_grid/sad_to_disgust.jpg" alt="Sad to Disgust"></td><td align="center"><img src="docs/demo_grid/sad_to_fear.jpg" alt="Sad to Fear"></td><td align="center"><img src="docs/demo_grid/sad_to_happy.jpg" alt="Sad to Happy"></td><td align="center" bgcolor="#fff7df"><img src="docs/demo_grid/sad_input.jpg" alt="Sad input"></td><td align="center"><img src="docs/demo_grid/sad_to_surprise.jpg" alt="Sad to Surprise"></td></tr>
<tr><td align="center"><b>256x256</b></td><td align="center"><img src="docs/demo_grid/surprise_to_neutral.jpg" alt="Surprise to Neutral"></td><td align="center"><img src="docs/demo_grid/surprise_to_angry.jpg" alt="Surprise to Angry"></td><td align="center"><img src="docs/demo_grid/surprise_to_disgust.jpg" alt="Surprise to Disgust"></td><td align="center"><img src="docs/demo_grid/surprise_to_fear.jpg" alt="Surprise to Fear"></td><td align="center"><img src="docs/demo_grid/surprise_to_happy.jpg" alt="Surprise to Happy"></td><td align="center"><img src="docs/demo_grid/surprise_to_sad.jpg" alt="Surprise to Sad"></td><td align="center" bgcolor="#fff7df"><img src="docs/demo_grid/surprise_input.jpg" alt="Surprise input"></td></tr>
</table>

<!-- demo-gallery-end -->
## 目录
- [样例结果](#样例结果)
- [1. 环境配置](#1-环境配置)
- [2. 数据准备](#2-数据准备)
  - [2.1 下载图像数据集 (FacePairEmoji)](#21-下载图像数据集-facepairemoji)
  - [2.2 生成表情配对 jsonl](#22-生成表情配对-jsonl)
  - [2.3 大模型质量过滤 + 指令生成](#23-大模型质量过滤--指令生成)
  - [2.4 离线提取 DECA 参数](#24-离线提取-deca-参数)
  - [2.5 校验提取结果](#25-校验提取结果)
- [3. Stage1 训练 (Conditional DECA Encoder)](#3-stage1-训练-conditional-deca-encoder)
- [4. Stage2 训练 (FLUX.2 Control Mixer)](#4-stage2-训练-flux2-control-mixer)
- [5. 推理](#5-推理)
  - [5.1 Stage1 推理 (控制图可视化)](#51-stage1-推理-控制图可视化)
  - [5.2 Stage2 推理 (端到端表情编辑)](#52-stage2-推理-端到端表情编辑)
- [6. 配置参数说明](#6-配置参数说明)
- [7. 项目结构](#7-项目结构)
- [致谢](#致谢)

---

## 1. 环境配置

推荐使用 `controlface310` conda 环境，CUDA 12.1 + PyTorch 2.5.1：

```bash
conda create -n controlface310 python=3.10 -y
conda activate controlface310

pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121

pip install -U setuptools wheel ninja cmake
conda install -y -c fvcore -c iopath -c conda-forge fvcore iopath
conda install -y -c conda-forge mpi4py dlib scikit-learn "scikit-image<0.25" tqdm

pip install -r requirements.txt
```

`set_env.sh` 已收录上述全部命令，可直接 `bash set_env.sh` 一键安装。

### PyTorch3D 安装

本仓库的 DECA renderer 默认使用 `pytorch3d` 后端，**强烈建议从源码本地安装**（`conda` channel 上的旧包与 torch 2.5 不兼容），实测可用版本是 `pytorch3d 0.7.9`：

```bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install --no-build-isolation -e .
```

`--no-build-isolation` 用于强制复用当前环境的 torch；否则 pip 可能会在隔离环境中报 `ModuleNotFoundError: No module named 'torch'`。

### 预训练权重

#### DECA Setup

在进行训练数据准备之前，需先下载 DECA 的源文件与权重完成安装（下载 FLAME 资源需要注册账号）：

1. `deca_model.tar`：访问 <https://github.com/YadiraF/DECA#:~:text=You%20can%20also%20use%20released%20model%20as%20pretrained%20model%2C%20then%20ignor%20the%20pretrain%20step.> 下载预训练的 DECA 权重。
2. `generic_model.pkl`：访问 <https://flame.is.tue.mpg.de/download.php> 下载 `FLAME 2020`，解压后取出 `generic_model.pkl`。
3. `FLAME_texture.npz`：访问 <https://flame.is.tue.mpg.de/download.php> 下载 `FLAME texture space`，解压后取出 `FLAME_texture.npz`。
4. 从 <https://github.com/YadiraF/DECA/tree/master/data> 下载下面其他文件，并一同放到 `data/` 目录下：

```bash
data/
  deca_model.tar
  generic_model.pkl
  FLAME_texture.npz
  fixed_displacement_256.npy
  head_template.obj
  landmark_embedding.npy
  mean_texture.jpg
  texture_data_256.npy
  uv_face_eye_mask.png
  uv_face_mask.png
```

#### FLUX.2-klein-base

从 HuggingFace 下载 `black-forest-labs/FLUX.2-klein-base-4B`（Diffusers 格式）到当前目录：

```bash
hf download black-forest-labs/FLUX.2-klein-base-4B \
    --local-dir ./flux-2-klein-base-4b
```

子目录 `text_encoder/` 与 `tokenizer/` 在 Stage1 中会被单独使用。下载完成后在 yaml 中通过 `flux.model_path` 指向本地路径（如 `./flux-2-klein-base-4b`）即可。

---

## 2. 数据准备

本仓库的训练样本是 **(参考图 A, 目标图 B, 文本指令)** 三元组。下面 5 步从「原始公开表情数据集」生成训练所需的 `*_pairs_with_instructions.jsonl` 与 `deca_params/*.pt`。

### 2.1 下载图像数据集 (FacePairEmoji)

图像数据已上传到 HuggingFace：[`yunpengZhangup/FacePairEmoji`](https://huggingface.co/datasets/yunpengZhangup/FacePairEmoji)。

```bash
hf download yunpengZhangup/FacePairEmoji --repo-type=dataset \
    --local-dir ./face_emoji
```

下载后建议得到如下两个根目录（路径名与本仓库 yaml 默认值对齐，强烈建议保持）：

```
./face_emoji/
├── final_data_raf_bucket_postprocessed/    # RAF-DB 经分桶 + 后处理
│   ├── 416x624/{neutral,angry,disgust,fear,happy,sad,surprise}/raf_xxx.png
│   └── 1040x1568/...
└── final_data_v1_bucket_postprocessed/     # KDEF / Multi-PIE / Oulu 等合集
    └── 544x736/{neutral,angry,...}/{kdef|multi_pie|oulu}_xxx.JPG
```

命名约定（决定后续 `extract_person_id` 行为）：

| 前缀 | 子集 | person_id 提取规则 |
|---|---|---|
| `raf_*` | RAF-DB | 文件名去扩展名整体作 person_id |
| `kdef_*` | KDEF | 取文件名中第 6-8 位 |
| `oulu_*` | Oulu-CASIA | 取下划线分割的第 1 段 |
| `multi_pie_*` | Multi-PIE | 取下划线分割的第 2 段 |

如果你使用自建数据集，只需按上面格式组织：`<root>/<bucket>/<expression>/<prefix>_<id>.<ext>`，并在 `scripts/generate_pairs_jsonl.py:extract_person_id` 中扩展前缀解析即可。

### 2.2 生成表情配对 jsonl

按 person_id 分组、对同一人的不同表情做 C(n,2) 全配对：

```bash
python scripts/generate_pairs_jsonl.py \
    --data_dir ./face_emoji/final_data_raf_bucket_postprocessed \
    --output   ./raf_pairs.jsonl

python scripts/generate_pairs_jsonl.py \
    --data_dir ./face_emoji/final_data_v1_bucket_postprocessed \
    --output   ./v1_pairs.jsonl
```

输出每行结构：

```json
{
  "pair_id": "kdef_F02_surprise_neutral", 
  "person_id": "kdef_F02", "dataset": "kdef", 
  "image_a_path": "./face_emoji/final_data_v1_bucket_postprocessed/surprise/kdef_AF02SUS.JPG", 
  "image_a_filename": "kdef_AF02SUS.JPG", 
  "expression_a": "surprise", 
  "image_b_path": "./face_emoji/final_data_v1_bucket_postprocessed/neutral/kdef_AF02NES.JPG", 
  "image_b_filename": "kdef_AF02NES.JPG", 
  "expression_b": "neutral",
  "check_result": null
}
```

> ⚠️ `image_*_path` 与 yaml 中 `data.sources[*].src_root` 必须使用同一种根目录写法。开源配置默认使用相对路径；如果你在本地改成绝对路径，请保持 jsonl 与 yaml 前缀一致，否则训练侧 `os.path.relpath(image_path, src_root)` 无法正确定位 `.pt` 参数。

### 2.3 大模型质量过滤 + 指令生成

本仓库**不提供具体的调用脚本**，仅公开这一阶段实际使用的 4 份 Prompt 模板。你可以把它们接到任意一家多模态 / 文本大模型 API 上（火山方舟 Doubao Vision、OpenAI GPT-4o、Gemini、Qwen-VL 等均可）复现同样的产物。

| 文件 | 用途 | 调用时需填充的变量 |
|---|---|---|
| [`prompts/check_og_data.txt`](prompts/check_og_data.txt) | **多模态质检**（同人判定 / 表情匹配 / 表情区分度 / 性别 / 质量），输出 JSON。 | `{expression_a}` `{expression_b}` |
| [`prompts/generate_instruction_en_sp.txt`](prompts/generate_instruction_en_sp.txt) | 生成**英文**编辑指令的 system prompt，输出严格 JSON 数组 (5 条) | —（仅作为 system） |
| [`prompts/generate_instruction_cn_sp.txt`](prompts/generate_instruction_cn_sp.txt) | 生成**中文**编辑指令的 system prompt，输出严格 JSON 数组 (5 条) | —（仅作为 system） |
| [`prompts/generate_instruction_user_prompt.txt`](prompts/generate_instruction_user_prompt.txt) | 上面两个指令生成 system prompt **配套的 user prompt** | `{expression_a}` `{expression_b}` `{gender}` |

最终输出的jsonl的每行结构如下：
```json
{
  "pair_id": "kdef_F02_surprise_neutral", 
  "person_id": "kdef_F02", 
  "dataset": "kdef", 
  "image_a_path": "./data/final_data_v1_bucket_postprocessed/544x736/surprise/kdef_AF02SUS.JPG", 
  "image_a_filename": "kdef_AF02SUS.JPG", 
  "expression_a": "surprise", 
  "image_b_path": "./data/final_data_v1_bucket_postprocessed/544x736/neutral/kdef_AF02NES.JPG", 
  "image_b_filename": "kdef_AF02NES.JPG", 
  "expression_b": "neutral", 
  "check_result": {
    "image_a_quality": "pass", "image_b_quality": "pass", "same_identity": true, "image_a_expression_match": true, "image_b_expression_match": true, "expression_distinguishable": true, "gender": "female", "overall": "pass", "reason": "1. 图片A质量观察：图片A包含清晰可辨的人脸，无严重遮挡、模糊、过曝或过暗问题，人脸为近似正脸，偏转角度远小于30°，质量合格。2. 图片B质量观察：图片B包含清晰可辨的人脸，无严重遮挡、模糊、过曝或过暗问题，人脸为近似正脸，偏转角度远小于30°，质量合格。3. 身份对比分析：两张图片的面部骨骼结构一致，脸型、颧骨、下颌线特征相同；五官特征匹配，眼睛形状、鼻子轮廓、嘴唇形态、眉形一致，面部痣的位置完全对应，发型、肤质也相同，判断为同一个人。4. 图片A表情匹配分析：图片A中人物眉毛高挑、额头有抬头纹、眼睛睁大、嘴巴张开呈O形，符合surprise（惊讶）表情的特征，与标注一致。5. 图片B表情匹配分析：图片B中人物面部肌肉放松，无明显表情动作，符合neutral（中性）表情的特征，与标注一致。6. 表情区分度分析：两张图片表情差异明显，A为夸张的惊讶表情，B为平静的中性表情，可清晰区分。7. 性别判断依据：人物面部轮廓柔和，眉形纤细，发型为女性常见的卷发，五官特征符合女性特点，判断为女性。"
    }, 
  "instruction_en": ["Make her face show a calm and relaxed expression.", "Turn her surprised look into a neutral one.", "Change her expression to a calm neutral state.", "Give her a soft relaxed facial expression.", "Let her face settle into a calm neutral look."], "instruction_cn": ["让她露出平静放松的神态。", "把她惊讶的神情变得淡然放松。", "将她的脸变为平和淡然的样子。", "使她呈现出平静放松的情绪。", "让画面中的人流露出淡然的神情。"]}
```

推荐流程：

1. 遵循 2.2 生成 `*_pairs.jsonl` 以后，在你自己的脚本里逐行读取 jsonl。
2. **质检**：调用多模态模型，把 `image_a_path` / `image_b_path` 两张图 + 填好变量后的 `check_og_data.txt` 一起送入，获得 `check_result` JSON。仅保留 `check_result.overall == "pass"` 的样本。
3. **指令生成**：对保留下来的样本调用两次纯文本 LLM：
   - 第 1 次：system = `generate_instruction_en_sp.txt`，user = 填好变量后的 `generate_instruction_user_prompt.txt`；返回严格 JSON 数组（5 条英文指令）存入 `instruction_en` 字段。
   - 第 2 次：system = `generate_instruction_cn_sp.txt`，user 同上；返回 5 条中文指令存入 `instruction_cn` 字段。
   - `gender` 可直接读 2 步产出的 `check_result.gender`。
4. 将上述产物写回为 `*_pairs_with_instructions.jsonl`，供后续 Stage1 / Stage2 训练使用。

### 2.4 离线提取 DECA 参数

训练时不会在线跑 DECA encode（太慢），而是预先把每张图的 DECA 编码结果离线保存为 `.pt`：

**单卡：**
```bash
python scripts/extract_deca_params.py \
    --src_root ./face_emoji/final_data_raf_bucket_postprocessed \
    --out_root ./deca_params/raf \
    --src_root ./face_emoji/final_data_v1_bucket_postprocessed \
    --out_root ./deca_params/v1 \
    --batch_size 32 --num_workers 16
```
**多卡分片（推荐）：**
```bash
  bash scripts/run_extract_multigpu.sh
```

输出目录与图像目录镜像对齐：

```
./face_emoji/final_data_raf_bucket_postprocessed/544x736/angry/raf_xxx.jpg
  ↓
./deca_params/raf/544x736/angry/raf_xxx.pt   # dict(shape, tex, exp, pose, cam, light, detail, tform)
```

断点续跑：脚本会自动跳过已存在的 `.pt`，多卡互相不重叠。失败样本写入 `{out_root}/_failed_shard{i}.jsonl`。

### 2.5 校验提取结果

```bash
python scripts/verify_deca_params.py \
    --src_root ./face_emoji/final_data_raf_bucket_postprocessed \
    --out_root ./deca_params/raf \
    --src_root ./face_emoji/final_data_v1_bucket_postprocessed \
    --out_root ./deca_params/v1 \
    --deep_check
```

会汇总：
- `_missing.txt`：源图存在但 `.pt` 缺失（需重跑 2.4）
- `_orphan.txt`：多余的 `.pt`（源图已删，可清理）
- `_dup_fail.txt`：既写入又出现在 failed 列表的异常样本
- `--deep_check` 还会逐个 `torch.load` 验证每个字段的 shape

---

## 3. Stage1 训练 (Conditional DECA Encoder)

Stage1 训练一个**文本条件的 DECA 表情/下颌头**：

- 输入：参考图 A（FAN crop 后 224×224）+ 文本指令 t
- 输出：预测目标表情参数 ψ_T (50 维 exp) 与 jaw_T (3 维 pose[3:6])
- 监督：以 image_b 离线提取的 DECA `exp[:50]` / `pose[3:6]` 为 GT，MSE + 正则
- 文本编码器：直接复用 FLUX.2-klein 的 Qwen3，concat 第 9/18/27 层 hidden（与官方 `encode_prompt` 完全一致）

配置文件：[`configs/stage1.yaml`](configs/stage1.yaml)

**启动：**
```bash
# 默认 8 卡 DDP
bash train/train_stage1.sh

日志写入 `logs/train_stage1/<timestamp>/train.log`，PID 写入同目录 `pid.txt`，停止：`kill $(cat logs/train_stage1/<ts>/pid.txt)`。

常用可调参数（更多见 yaml 注释）：

| 字段 | 默认 | 说明 |
|---|---|---|
| `data.sources[*].jsonl/src_root/params_root` | – | 多个数据源会自动合并 |
| `data.lang_prob_en` | 0.5 | 中/英指令抽样概率 |
| `model.text_encoder_path` | FLUX.2-klein 的 `text_encoder/` | Qwen3 权重目录 |
| `model.deca_model_tar` | `./data/deca_model.tar` | DECA ResNet50 backbone |
| `train.epochs / batch_size / lr / warmup_ratio` | 10 / 32 / 1e-4 / 0.05 | 优化器配置 |
| `train.val_every_steps` | 20 | 每多少步跑一次验证（同时触发 best ckpt 保存） |
| `ckpt.save_best_only` | true | 仅保留 `best-step-{N}.pt` + `last.pt` |

**输出：** `./checkpoints/stage1/stage1-<timestamp>/best-step-{N}.pt`，将作为 Stage2 的初始权重。

---

## 4. Stage2 训练 (FLUX.2 Control Mixer)

Stage2 在 Stage1 基础上联合训练：

1. **DECA 渲染** 把 Stage1 预测的 (ψ_T, jaw_T) 与参考图的 (shape/tex/cam/light) 组合，渲染出目标控制图 D_T = (rendered, normal, albedo, 共 9ch)；参考图本身的 D_R 同样在线渲染
2. **Flux2ControlMixer (CMM)** 把 (D_T, D_R) 投影成 token 注入 FLUX.2 transformer
3. **Flow matching loss** 在 FLUX.2 latent 空间监督，加一份 Stage1 的 aux loss 维持表情一致性
4. **DDP + BucketBatchSampler** 按图像分辨率分桶，保证同 batch 内尺寸一致

配置文件：[`configs/stage2.yaml`](configs/stage2.yaml)

**启动：**
```bash
# 默认 8 卡 DDP
bash train/train_stage2.sh


关键配置（详见 yaml）：

| 字段 | 默认 | 说明 |
|---|---|---|
| `stage1.ckpt_path` | `./checkpoints/stage1/.../best-step-2320.pt` | **必须**指向上一步训练好的 Stage1 ckpt |
| `stage2.resume_path` | null | 若中断，可填某 ckpt 续训 |
| `stage2.detach_deca_control` | true | true 时 flow loss 不反传 DECA renderer，省显存；false 走完整端到端 |
| `flux.model_path` | FLUX.2-klein-base 目录 | 需 Diffusers 格式；klein-base **非蒸馏**，transformer guidance 必须传 None |
| `model.control_mixer_*` | 512 / 8 / 128 | CMM 结构（推理 yaml 必须与训练保持一致） |
| `data.use_alpha_mask` | true | 控制图是否用 alpha mask 把背景置 0 |
| `loss.lambda_flow / lambda_aux` | 1.0 / 0.1 | flow loss 与 Stage1 aux loss 的权重 |
| `train.gradient_checkpointing` | true | 推荐开启，FLUX.2 单卡 80G 才放得下 batch=1 |

**输出：** `./checkpoints/stage2/stage2-<timestamp>/best-step-{N}.pt`，包含 `{stage1_model, control_mixer, cfg, step, ...}`，是推理时唯一需要的权重。

---

## 5. 推理

### 5.1 Stage1 推理 (控制图可视化)

用于验证 Stage1 是否能根据 prompt 输出合理的表情几何，输出 6 张图（参考路径 D_R + 目标路径 D_T，各 rendered/normal/albedo）+ 2 个 9ch tensor。

```bash
python infer/infer_stage1.py \
    --ref     ./原图.png \
    --prompt  "make her burst into laughter" \
    --output_dir ./output_stage1 \
    --ckpt    ./checkpoints/stage1/stage1-<timestamp>/best-step-2320.pt
```

### 5.2 Stage2 推理 (端到端表情编辑)

配置文件：[`configs/infer_stage2.yaml`](configs/infer_stage2.yaml) （独立于训练 yaml，只放推理相关字段）。

**默认用法（启用 RCG，λ=3.0 从 yaml 读取）：**
```bash
python infer/infer_stage2.py \
    --config       configs/infer_stage2.yaml \
    --ref          原图.png \
    --prompt       "make her burst into laughter" \
    --stage2_ckpt  ./checkpoints/stage2/stage2-<timestamp>/best-step-3480.pt \
    --output_dir   ./output_stage2
```

**Sweep RCG 系数 λ（推荐 0/1/3/5/7）：**
```bash
for LAM in 0.0 1.0 3.0 5.0 7.0; do
  python infer/infer_stage2.py \
    --config configs/infer_stage2.yaml \
    --ref 原图.png \
    --prompt "make her burst into laughter" \
    --stage2_ckpt ./checkpoints/stage2/.../best-step-1920.pt \
    --rcg_lambda ${LAM} \
    --output_dir ./output_stage2/sweep_lambda_${LAM}
done
```

**关闭 RCG 做对照：**
```bash
python infer/infer_stage2.py \
    --config configs/infer_stage2.yaml \
    --ref 原图.png \
    --prompt "make her burst into laughter" \
    --stage2_ckpt ./checkpoints/stage2/.../best-step-1920.pt \
    --rcg_enabled false \
    --output_dir ./output_stage2/no_rcg
```

**临时覆盖 yaml 任意字段：**
```bash
python infer/infer_stage2.py --config configs/infer_stage2.yaml \
    --ref 原图.png --prompt "smile" \
    --stage2_ckpt ./checkpoints/stage2/.../best-step-1920.pt \
    --opts sampling.num_inference_steps=28 sampling.height=512 sampling.width=512 \
           sampling.rcg.lambda=2.5
```

**RCG (Reference Control Guidance) 简介：** 在 CMM 的两路条件之间外推：

```
eps_ref = DiT(cat(latents, ref_latents, CMM(D_R, D_R)), text)   # ref 路径
eps_tgt = DiT(cat(latents, ref_latents, CMM(D_T, D_R)), text)   # tgt 路径
eps     = eps_ref + λ · (eps_tgt - eps_ref)
```

| λ | 行为 |
|---|---|
| 0.0 | 生成 ≈ 参考图（几乎不变） |
| 1.0 | 等价于无 RCG（eps = eps_tgt） |
| 3.0 | 默认，表情清晰且身份稳 |
| >3 | 表情更夸张，但身份漂移风险上升 |

推理输出（默认 `output_stage2/`）：
- `final.png`：FLUX.2 生成的最终图
- `D_R_*.png` / `D_T_*.png`：参考/目标控制图
- `*_9ch.pt` / `*_tokens.pt`：调试中间张量（`output.save_intermediates=true` 才生成）
- `summary.json`：本次推理的所有有效配置

---

## 6. 配置参数说明

本仓库共有三个 yaml 配置文件，分别对应训练 / 推理的不同阶段，所有可调字段都已在文件中带详细中文注释，按需修改即可：

| 配置文件 | 作用 |
|---|---|
| [`configs/stage1.yaml`](configs/stage1.yaml) | Stage1 训练：Conditional DECA Encoder 的数据源 / 模型结构 / 优化器 / loss / ckpt 策略 |
| [`configs/stage2.yaml`](configs/stage2.yaml) | Stage2 训练：FLUX.2 Control Mixer 的端到端联合训练，引用 Stage1 ckpt + FLUX.2 主干 + CMM 结构 |
| [`configs/infer_stage2.yaml`](configs/infer_stage2.yaml) | Stage2 推理：FLUX.2 采样参数 + RCG 系数 + Stage2 ckpt 路径，结构字段须与 stage2.yaml 保持一致 |

所有字段也支持通过 `--opts key=val` 在 CLI 临时覆盖，无需修改文件。

---

## 7. 项目结构

```
ControlFace-main/
├── configs/                       # 所有 yaml 配置
│   ├── stage1.yaml                # Stage1 训练配置
│   ├── stage2.yaml                # Stage2 训练配置
│   └── infer_stage2.yaml          # Stage2 推理配置
├── data/                          # DECA 静态资产 (head_template.obj / mask 等)
├── decalib/                       # DECA 官方代码 (encoder / FLAME / renderer)
├── deca_params/                   # 离线提取的 .pt (步骤 2.4 产物)
│   ├── raf/<bucket>/<expr>/*.pt
│   └── v1/<bucket>/<expr>/*.pt
├── infer/
│   ├── infer_stage1.py            # 控制图可视化
│   └── infer_stage2.py            # 端到端表情编辑
├── prompts/check_og_data.txt      # 火山方舟质检 prompt 模板
├── scripts/
│   ├── generate_pairs_jsonl.py    # 步骤 2.2: 生成配对 jsonl
│   ├── check_faces.py             # 步骤 2.3: 大模型质量过滤 + 指令生成
│   ├── extract_deca_params.py     # 步骤 2.4: 单卡 DECA 提取
│   ├── run_extract_multigpu.sh    # 步骤 2.4: 多卡分片
│   └── verify_deca_params.py      # 步骤 2.5: 完整性校验
├── src/
│   ├── datasets/                  # Stage1 / Stage2 Dataset + collate + BucketSampler
│   ├── losses/                    # Stage1 / Stage2 loss
│   └── models/                    # ConditionalDECAEncoder / Flux2ControlMixer
├── train/
│   ├── train_stage1.{py,sh}
│   └── train_stage2.{py,sh}
├── raf_pairs_with_instructions.jsonl   # 步骤 2.3 产物
├── v1_pairs_with_instructions.jsonl    # 步骤 2.3 产物
├── requirements.txt
├── set_env.sh                     # 一键安装
└── README.md
```

---

## 致谢

本项目在以下工作的基础上构建：
- [ControlFace (CVPR 2025)](https://github.com/cvlab-kaist/ControlFace) – 控制范式与整体架构
- [DECA](https://github.com/yfeng95/DECA) / [DiffusionRig](https://github.com/adobe-research/diffusion-rig) – 3D 人脸几何编码与渲染
- [FLUX.2-klein-base](https://huggingface.co/black-forest-labs/FLUX.2-klein-base) – 主干流匹配模型
- [face-alignment (FAN)](https://github.com/1adrianb/face-alignment) – 在线人脸关键点检测

感谢上游作者们开源的工作。
