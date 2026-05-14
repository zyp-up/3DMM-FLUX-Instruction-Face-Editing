<p align="center">
  <h1 align="center">3DMM-FLUX: Instruction-Guided Facial Expression Editing</h1>
  <h1 align="center">基于 3DMM 参数化表征与 FLUX 扩散模型的指令驱动人脸表情编辑框架</h1>
</p>

<p align="center">
  <img src="docs/figures/overview.png" width="960" alt="3DMM-FLUX overall architecture">
</p>

<p align="center">
  <a href="#zh">Chinese</a> | <a href="#en">English</a>
</p>

<a id="zh"></a>

## Abstract

本项目实现了一套基于 3DMM 参数化表征与 FLUX 扩散模型的指令驱动人脸表情编辑框架。给定单张人脸图像与自然语言指令，系统首先利用 DECA/3DMM 显式建模输入人脸的身份几何、姿态与表情参数，再预测目标表情对应的参数化几何控制信号，并将该控制信号注入 FLUX.2-klein 的流匹配生成过程。该设计把可解释的三维人脸先验与高保真生成模型结合起来，在增强目标表情可控性的同时尽量保持原始身份特征一致。

本仓库基于 [ControlFace (CVPR 2025)](https://github.com/cvlab-kaist/ControlFace) 的控制思想、[DECA](https://github.com/yfeng95/DECA) 的 3DMM 参数估计与渲染能力，以及 [FLUX.2-klein-base](https://huggingface.co/black-forest-labs/FLUX.2-klein-base) 的官方文本编码器和流匹配生成配置完成训练与推理实现。训练数据使用自建的 FacePairEmoji 表情配对数据集，并支持中文与英文编辑指令。

## Highlights

- **参数化表情控制**：使用 DECA/3DMM 将输入人脸分解为身份、姿态、表情、光照和相机参数，避免仅依赖隐式图像条件。
- **指令驱动编辑**：支持中文和英文自然语言指令，预测目标表情的 DECA 表情参数与下颌姿态。
- **FLUX.2 流匹配生成**：将目标与参考 3DMM 控制图编码为控制 token，注入 FLUX.2-klein 生成主干。
- **身份保持约束**：通过参考控制路径与 RCG 推理策略增强身份一致性，并与原始 FLUX.2 基线进行并列对比。

<!-- demo-gallery-start -->
## Demo

下表以源表情和目标表情构成 7×7 定性对比矩阵，行列均按 `Neutral → Angry → Disgust → Fear → Happy → Sad → Surprise` 排列。每一行对应同一输入身份，左侧仅标注该样例原始分辨率；浅黄色对角线单元格为原始输入图，其余单元格直接引用 `demo_output/` 中的原始生成结果。每个非对角单元格使用嵌套表格展示四图对比：上排为本方法 `Control-CN / Control-EN`，下排为无 3DMM 控制的原始 FLUX.2 基线 `OG-CN / OG-EN`。

<table>
<tr><th>Resolution</th><th>Neutral</th><th>Angry</th><th>Disgust</th><th>Fear</th><th>Happy</th><th>Sad</th><th>Surprise</th></tr>
<tr><td align="center"><b>2048x2048</b></td><td align="center" bgcolor="#fff7df"><b>INPUT</b><br><img src="demo_input/2048*2048_neutral/raf_train_12225.jpg" width="220" alt="Neutral input"><br><sub>Neutral</sub></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/2048*2048_neutral__raf_train_12225/angry_cn.png" width="105" alt="Neutral to Angry Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/2048*2048_neutral__raf_train_12225/angry_en.png" width="105" alt="Neutral to Angry Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/2048*2048_neutral__raf_train_12225/angry_cn.png" width="105" alt="Neutral to Angry OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/2048*2048_neutral__raf_train_12225/angry_en.png" width="105" alt="Neutral to Angry OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/2048*2048_neutral__raf_train_12225/disgust_cn.png" width="105" alt="Neutral to Disgust Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/2048*2048_neutral__raf_train_12225/disgust_en.png" width="105" alt="Neutral to Disgust Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/2048*2048_neutral__raf_train_12225/disgust_cn.png" width="105" alt="Neutral to Disgust OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/2048*2048_neutral__raf_train_12225/disgust_en.png" width="105" alt="Neutral to Disgust OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/2048*2048_neutral__raf_train_12225/fear_cn.png" width="105" alt="Neutral to Fear Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/2048*2048_neutral__raf_train_12225/fear_en.png" width="105" alt="Neutral to Fear Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/2048*2048_neutral__raf_train_12225/fear_cn.png" width="105" alt="Neutral to Fear OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/2048*2048_neutral__raf_train_12225/fear_en.png" width="105" alt="Neutral to Fear OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/2048*2048_neutral__raf_train_12225/happy_cn.png" width="105" alt="Neutral to Happy Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/2048*2048_neutral__raf_train_12225/happy_en.png" width="105" alt="Neutral to Happy Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/2048*2048_neutral__raf_train_12225/happy_cn.png" width="105" alt="Neutral to Happy OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/2048*2048_neutral__raf_train_12225/happy_en.png" width="105" alt="Neutral to Happy OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/2048*2048_neutral__raf_train_12225/sad_cn.png" width="105" alt="Neutral to Sad Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/2048*2048_neutral__raf_train_12225/sad_en.png" width="105" alt="Neutral to Sad Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/2048*2048_neutral__raf_train_12225/sad_cn.png" width="105" alt="Neutral to Sad OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/2048*2048_neutral__raf_train_12225/sad_en.png" width="105" alt="Neutral to Sad OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/2048*2048_neutral__raf_train_12225/surprise_cn.png" width="105" alt="Neutral to Surprise Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/2048*2048_neutral__raf_train_12225/surprise_en.png" width="105" alt="Neutral to Surprise Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/2048*2048_neutral__raf_train_12225/surprise_cn.png" width="105" alt="Neutral to Surprise OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/2048*2048_neutral__raf_train_12225/surprise_en.png" width="105" alt="Neutral to Surprise OG-EN"></td></tr></table></td></tr>
<tr><td align="center"><b>1328x1776</b></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/1328*1776_angry__raf_train_09783/neutral_cn.png" width="105" alt="Angry to Neutral Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/1328*1776_angry__raf_train_09783/neutral_en.png" width="105" alt="Angry to Neutral Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/1328*1776_angry__raf_train_09783/neutral_cn.png" width="105" alt="Angry to Neutral OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/1328*1776_angry__raf_train_09783/neutral_en.png" width="105" alt="Angry to Neutral OG-EN"></td></tr></table></td><td align="center" bgcolor="#fff7df"><b>INPUT</b><br><img src="demo_input/1328*1776_angry/raf_train_09783.png" width="220" alt="Angry input"><br><sub>Angry</sub></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/1328*1776_angry__raf_train_09783/disgust_cn.png" width="105" alt="Angry to Disgust Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/1328*1776_angry__raf_train_09783/disgust_en.png" width="105" alt="Angry to Disgust Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/1328*1776_angry__raf_train_09783/disgust_cn.png" width="105" alt="Angry to Disgust OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/1328*1776_angry__raf_train_09783/disgust_en.png" width="105" alt="Angry to Disgust OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/1328*1776_angry__raf_train_09783/fear_cn.png" width="105" alt="Angry to Fear Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/1328*1776_angry__raf_train_09783/fear_en.png" width="105" alt="Angry to Fear Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/1328*1776_angry__raf_train_09783/fear_cn.png" width="105" alt="Angry to Fear OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/1328*1776_angry__raf_train_09783/fear_en.png" width="105" alt="Angry to Fear OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/1328*1776_angry__raf_train_09783/happy_cn.png" width="105" alt="Angry to Happy Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/1328*1776_angry__raf_train_09783/happy_en.png" width="105" alt="Angry to Happy Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/1328*1776_angry__raf_train_09783/happy_cn.png" width="105" alt="Angry to Happy OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/1328*1776_angry__raf_train_09783/happy_en.png" width="105" alt="Angry to Happy OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/1328*1776_angry__raf_train_09783/sad_cn.png" width="105" alt="Angry to Sad Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/1328*1776_angry__raf_train_09783/sad_en.png" width="105" alt="Angry to Sad Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/1328*1776_angry__raf_train_09783/sad_cn.png" width="105" alt="Angry to Sad OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/1328*1776_angry__raf_train_09783/sad_en.png" width="105" alt="Angry to Sad OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/1328*1776_angry__raf_train_09783/surprise_cn.png" width="105" alt="Angry to Surprise Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/1328*1776_angry__raf_train_09783/surprise_en.png" width="105" alt="Angry to Surprise Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/1328*1776_angry__raf_train_09783/surprise_cn.png" width="105" alt="Angry to Surprise OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/1328*1776_angry__raf_train_09783/surprise_en.png" width="105" alt="Angry to Surprise OG-EN"></td></tr></table></td></tr>
<tr><td align="center"><b>1184x1392</b></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/1184*1392_disgust__raf_train_11947/neutral_cn.png" width="105" alt="Disgust to Neutral Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/1184*1392_disgust__raf_train_11947/neutral_en.png" width="105" alt="Disgust to Neutral Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/1184*1392_disgust__raf_train_11947/neutral_cn.png" width="105" alt="Disgust to Neutral OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/1184*1392_disgust__raf_train_11947/neutral_en.png" width="105" alt="Disgust to Neutral OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/1184*1392_disgust__raf_train_11947/angry_cn.png" width="105" alt="Disgust to Angry Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/1184*1392_disgust__raf_train_11947/angry_en.png" width="105" alt="Disgust to Angry Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/1184*1392_disgust__raf_train_11947/angry_cn.png" width="105" alt="Disgust to Angry OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/1184*1392_disgust__raf_train_11947/angry_en.png" width="105" alt="Disgust to Angry OG-EN"></td></tr></table></td><td align="center" bgcolor="#fff7df"><b>INPUT</b><br><img src="demo_input/1184*1392_disgust/raf_train_11947.png" width="220" alt="Disgust input"><br><sub>Disgust</sub></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/1184*1392_disgust__raf_train_11947/fear_cn.png" width="105" alt="Disgust to Fear Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/1184*1392_disgust__raf_train_11947/fear_en.png" width="105" alt="Disgust to Fear Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/1184*1392_disgust__raf_train_11947/fear_cn.png" width="105" alt="Disgust to Fear OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/1184*1392_disgust__raf_train_11947/fear_en.png" width="105" alt="Disgust to Fear OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/1184*1392_disgust__raf_train_11947/happy_cn.png" width="105" alt="Disgust to Happy Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/1184*1392_disgust__raf_train_11947/happy_en.png" width="105" alt="Disgust to Happy Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/1184*1392_disgust__raf_train_11947/happy_cn.png" width="105" alt="Disgust to Happy OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/1184*1392_disgust__raf_train_11947/happy_en.png" width="105" alt="Disgust to Happy OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/1184*1392_disgust__raf_train_11947/sad_cn.png" width="105" alt="Disgust to Sad Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/1184*1392_disgust__raf_train_11947/sad_en.png" width="105" alt="Disgust to Sad Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/1184*1392_disgust__raf_train_11947/sad_cn.png" width="105" alt="Disgust to Sad OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/1184*1392_disgust__raf_train_11947/sad_en.png" width="105" alt="Disgust to Sad OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/1184*1392_disgust__raf_train_11947/surprise_cn.png" width="105" alt="Disgust to Surprise Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/1184*1392_disgust__raf_train_11947/surprise_en.png" width="105" alt="Disgust to Surprise Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/1184*1392_disgust__raf_train_11947/surprise_cn.png" width="105" alt="Disgust to Surprise OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/1184*1392_disgust__raf_train_11947/surprise_en.png" width="105" alt="Disgust to Surprise OG-EN"></td></tr></table></td></tr>
<tr><td align="center"><b>832x1248</b></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/832*1248_fear__raf_test_2428/neutral_cn.png" width="105" alt="Fear to Neutral Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/832*1248_fear__raf_test_2428/neutral_en.png" width="105" alt="Fear to Neutral Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/832*1248_fear__raf_test_2428/neutral_cn.png" width="105" alt="Fear to Neutral OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/832*1248_fear__raf_test_2428/neutral_en.png" width="105" alt="Fear to Neutral OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/832*1248_fear__raf_test_2428/angry_cn.png" width="105" alt="Fear to Angry Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/832*1248_fear__raf_test_2428/angry_en.png" width="105" alt="Fear to Angry Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/832*1248_fear__raf_test_2428/angry_cn.png" width="105" alt="Fear to Angry OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/832*1248_fear__raf_test_2428/angry_en.png" width="105" alt="Fear to Angry OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/832*1248_fear__raf_test_2428/disgust_cn.png" width="105" alt="Fear to Disgust Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/832*1248_fear__raf_test_2428/disgust_en.png" width="105" alt="Fear to Disgust Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/832*1248_fear__raf_test_2428/disgust_cn.png" width="105" alt="Fear to Disgust OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/832*1248_fear__raf_test_2428/disgust_en.png" width="105" alt="Fear to Disgust OG-EN"></td></tr></table></td><td align="center" bgcolor="#fff7df"><b>INPUT</b><br><img src="demo_input/832*1248_fear/raf_test_2428.png" width="220" alt="Fear input"><br><sub>Fear</sub></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/832*1248_fear__raf_test_2428/happy_cn.png" width="105" alt="Fear to Happy Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/832*1248_fear__raf_test_2428/happy_en.png" width="105" alt="Fear to Happy Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/832*1248_fear__raf_test_2428/happy_cn.png" width="105" alt="Fear to Happy OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/832*1248_fear__raf_test_2428/happy_en.png" width="105" alt="Fear to Happy OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/832*1248_fear__raf_test_2428/sad_cn.png" width="105" alt="Fear to Sad Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/832*1248_fear__raf_test_2428/sad_en.png" width="105" alt="Fear to Sad Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/832*1248_fear__raf_test_2428/sad_cn.png" width="105" alt="Fear to Sad OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/832*1248_fear__raf_test_2428/sad_en.png" width="105" alt="Fear to Sad OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/832*1248_fear__raf_test_2428/surprise_cn.png" width="105" alt="Fear to Surprise Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/832*1248_fear__raf_test_2428/surprise_en.png" width="105" alt="Fear to Surprise Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/832*1248_fear__raf_test_2428/surprise_cn.png" width="105" alt="Fear to Surprise OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/832*1248_fear__raf_test_2428/surprise_en.png" width="105" alt="Fear to Surprise OG-EN"></td></tr></table></td></tr>
<tr><td align="center"><b>656x896</b></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/656*896_happy__raf_train_10863/neutral_cn.png" width="105" alt="Happy to Neutral Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/656*896_happy__raf_train_10863/neutral_en.png" width="105" alt="Happy to Neutral Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/656*896_happy__raf_train_10863/neutral_cn.png" width="105" alt="Happy to Neutral OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/656*896_happy__raf_train_10863/neutral_en.png" width="105" alt="Happy to Neutral OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/656*896_happy__raf_train_10863/angry_cn.png" width="105" alt="Happy to Angry Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/656*896_happy__raf_train_10863/angry_en.png" width="105" alt="Happy to Angry Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/656*896_happy__raf_train_10863/angry_cn.png" width="105" alt="Happy to Angry OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/656*896_happy__raf_train_10863/angry_en.png" width="105" alt="Happy to Angry OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/656*896_happy__raf_train_10863/disgust_cn.png" width="105" alt="Happy to Disgust Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/656*896_happy__raf_train_10863/disgust_en.png" width="105" alt="Happy to Disgust Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/656*896_happy__raf_train_10863/disgust_cn.png" width="105" alt="Happy to Disgust OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/656*896_happy__raf_train_10863/disgust_en.png" width="105" alt="Happy to Disgust OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/656*896_happy__raf_train_10863/fear_cn.png" width="105" alt="Happy to Fear Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/656*896_happy__raf_train_10863/fear_en.png" width="105" alt="Happy to Fear Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/656*896_happy__raf_train_10863/fear_cn.png" width="105" alt="Happy to Fear OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/656*896_happy__raf_train_10863/fear_en.png" width="105" alt="Happy to Fear OG-EN"></td></tr></table></td><td align="center" bgcolor="#fff7df"><b>INPUT</b><br><img src="demo_input/656*896_happy/raf_train_10863.png" width="220" alt="Happy input"><br><sub>Happy</sub></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/656*896_happy__raf_train_10863/sad_cn.png" width="105" alt="Happy to Sad Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/656*896_happy__raf_train_10863/sad_en.png" width="105" alt="Happy to Sad Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/656*896_happy__raf_train_10863/sad_cn.png" width="105" alt="Happy to Sad OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/656*896_happy__raf_train_10863/sad_en.png" width="105" alt="Happy to Sad OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/656*896_happy__raf_train_10863/surprise_cn.png" width="105" alt="Happy to Surprise Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/656*896_happy__raf_train_10863/surprise_en.png" width="105" alt="Happy to Surprise Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/656*896_happy__raf_train_10863/surprise_cn.png" width="105" alt="Happy to Surprise OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/656*896_happy__raf_train_10863/surprise_en.png" width="105" alt="Happy to Surprise OG-EN"></td></tr></table></td></tr>
<tr><td align="center"><b>448x592</b></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/448*592_sad__raf_train_10607/neutral_cn.png" width="105" alt="Sad to Neutral Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/448*592_sad__raf_train_10607/neutral_en.png" width="105" alt="Sad to Neutral Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/448*592_sad__raf_train_10607/neutral_cn.png" width="105" alt="Sad to Neutral OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/448*592_sad__raf_train_10607/neutral_en.png" width="105" alt="Sad to Neutral OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/448*592_sad__raf_train_10607/angry_cn.png" width="105" alt="Sad to Angry Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/448*592_sad__raf_train_10607/angry_en.png" width="105" alt="Sad to Angry Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/448*592_sad__raf_train_10607/angry_cn.png" width="105" alt="Sad to Angry OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/448*592_sad__raf_train_10607/angry_en.png" width="105" alt="Sad to Angry OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/448*592_sad__raf_train_10607/disgust_cn.png" width="105" alt="Sad to Disgust Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/448*592_sad__raf_train_10607/disgust_en.png" width="105" alt="Sad to Disgust Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/448*592_sad__raf_train_10607/disgust_cn.png" width="105" alt="Sad to Disgust OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/448*592_sad__raf_train_10607/disgust_en.png" width="105" alt="Sad to Disgust OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/448*592_sad__raf_train_10607/fear_cn.png" width="105" alt="Sad to Fear Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/448*592_sad__raf_train_10607/fear_en.png" width="105" alt="Sad to Fear Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/448*592_sad__raf_train_10607/fear_cn.png" width="105" alt="Sad to Fear OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/448*592_sad__raf_train_10607/fear_en.png" width="105" alt="Sad to Fear OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/448*592_sad__raf_train_10607/happy_cn.png" width="105" alt="Sad to Happy Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/448*592_sad__raf_train_10607/happy_en.png" width="105" alt="Sad to Happy Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/448*592_sad__raf_train_10607/happy_cn.png" width="105" alt="Sad to Happy OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/448*592_sad__raf_train_10607/happy_en.png" width="105" alt="Sad to Happy OG-EN"></td></tr></table></td><td align="center" bgcolor="#fff7df"><b>INPUT</b><br><img src="demo_input/448*592_sad/raf_train_10607.png" width="220" alt="Sad input"><br><sub>Sad</sub></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/448*592_sad__raf_train_10607/surprise_cn.png" width="105" alt="Sad to Surprise Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/448*592_sad__raf_train_10607/surprise_en.png" width="105" alt="Sad to Surprise Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/448*592_sad__raf_train_10607/surprise_cn.png" width="105" alt="Sad to Surprise OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/448*592_sad__raf_train_10607/surprise_en.png" width="105" alt="Sad to Surprise OG-EN"></td></tr></table></td></tr>
<tr><td align="center"><b>256x256</b></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/256*256_surprise__raf_train_10109/neutral_cn.png" width="105" alt="Surprise to Neutral Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/256*256_surprise__raf_train_10109/neutral_en.png" width="105" alt="Surprise to Neutral Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/256*256_surprise__raf_train_10109/neutral_cn.png" width="105" alt="Surprise to Neutral OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/256*256_surprise__raf_train_10109/neutral_en.png" width="105" alt="Surprise to Neutral OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/256*256_surprise__raf_train_10109/angry_cn.png" width="105" alt="Surprise to Angry Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/256*256_surprise__raf_train_10109/angry_en.png" width="105" alt="Surprise to Angry Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/256*256_surprise__raf_train_10109/angry_cn.png" width="105" alt="Surprise to Angry OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/256*256_surprise__raf_train_10109/angry_en.png" width="105" alt="Surprise to Angry OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/256*256_surprise__raf_train_10109/disgust_cn.png" width="105" alt="Surprise to Disgust Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/256*256_surprise__raf_train_10109/disgust_en.png" width="105" alt="Surprise to Disgust Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/256*256_surprise__raf_train_10109/disgust_cn.png" width="105" alt="Surprise to Disgust OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/256*256_surprise__raf_train_10109/disgust_en.png" width="105" alt="Surprise to Disgust OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/256*256_surprise__raf_train_10109/fear_cn.png" width="105" alt="Surprise to Fear Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/256*256_surprise__raf_train_10109/fear_en.png" width="105" alt="Surprise to Fear Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/256*256_surprise__raf_train_10109/fear_cn.png" width="105" alt="Surprise to Fear OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/256*256_surprise__raf_train_10109/fear_en.png" width="105" alt="Surprise to Fear OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/256*256_surprise__raf_train_10109/happy_cn.png" width="105" alt="Surprise to Happy Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/256*256_surprise__raf_train_10109/happy_en.png" width="105" alt="Surprise to Happy Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/256*256_surprise__raf_train_10109/happy_cn.png" width="105" alt="Surprise to Happy OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/256*256_surprise__raf_train_10109/happy_en.png" width="105" alt="Surprise to Happy OG-EN"></td></tr></table></td><td align="center"><table><tr><td align="center"><sub>Control-CN</sub><br><img src="demo_output/control_flux/256*256_surprise__raf_train_10109/sad_cn.png" width="105" alt="Surprise to Sad Control-CN"></td><td align="center"><sub>Control-EN</sub><br><img src="demo_output/control_flux/256*256_surprise__raf_train_10109/sad_en.png" width="105" alt="Surprise to Sad Control-EN"></td></tr><tr><td align="center"><sub>OG-CN</sub><br><img src="demo_output/og_flux/256*256_surprise__raf_train_10109/sad_cn.png" width="105" alt="Surprise to Sad OG-CN"></td><td align="center"><sub>OG-EN</sub><br><img src="demo_output/og_flux/256*256_surprise__raf_train_10109/sad_en.png" width="105" alt="Surprise to Sad OG-EN"></td></tr></table></td><td align="center" bgcolor="#fff7df"><b>INPUT</b><br><img src="demo_input/256*256_surprise/raf_train_10109.png" width="220" alt="Surprise input"><br><sub>Surprise</sub></td></tr>
</table>

<!-- demo-gallery-end -->

## 1. Environment

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

### PyTorch3D Installation

本仓库的 DECA renderer 默认使用 `pytorch3d` 后端，**强烈建议从源码本地安装**（`conda` channel 上的旧包与 torch 2.5 不兼容），实测可用版本是 `pytorch3d 0.7.9`：

```bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install --no-build-isolation -e .
```

`--no-build-isolation` 用于强制复用当前环境的 torch；否则 pip 可能会在隔离环境中报 `ModuleNotFoundError: No module named 'torch'`。

### DECA Setup

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

### FLUX.2-klein-base

从 HuggingFace 下载 `black-forest-labs/FLUX.2-klein-base-4B`（Diffusers 格式）到当前目录：

```bash
hf download black-forest-labs/FLUX.2-klein-base-4B \
    --local-dir ./flux-2-klein-base-4b
```

子目录 `text_encoder/` 与 `tokenizer/` 在 Stage1 中会被单独使用。下载完成后在 yaml 中通过 `flux.model_path` 指向本地路径（如 `./flux-2-klein-base-4b`）即可。

---

## 2. Data Preparation

本仓库的训练样本是 **(参考图 A, 目标图 B, 文本指令)** 三元组。你可以直接使用我们在 Hugging Face 上发布的预处理版本，其中已经包含图像、`*_pairs_with_instructions.jsonl` 和预提取的 `deca_params/*.pt`；也可以使用自己的数据集，并按 2.2–2.5 的流程重新生成 jsonl、指令与 DECA 参数。

### 2.1 Download Preprocessed Dataset (FacePairEmoji)

预处理数据已上传到 Hugging Face：[`yunpengZhangup/FacePairEmoji`](https://huggingface.co/datasets/yunpengZhangup/FacePairEmoji)。仓库中包含：

- `final_data_raf_bucket_postprocessed/`：RAF-DB 分桶后图像。
- `final_data_v1_bucket_postprocessed/`：KDEF / Multi-PIE / Oulu 等合集分桶后图像。
- `raf_pairs_with_instructions.jsonl` 与 `v1_pairs_with_instructions.jsonl`：已质检并生成中英文指令的训练配对。
- `deca_params/`：与图像目录镜像对齐的预提取 DECA 参数。

```bash
hf download yunpengZhangup/FacePairEmoji --repo-type=dataset \
    --local-dir ./face_emoji
```

下载后目录应与本仓库默认配置对齐：

```
./face_emoji/
├── raf_pairs_with_instructions.jsonl
├── v1_pairs_with_instructions.jsonl
├── deca_params/
│   ├── raf/
│   └── v1/
├── final_data_raf_bucket_postprocessed/    # RAF-DB 经分桶 + 后处理
│   ├── 416x624/{neutral,angry,disgust,fear,happy,sad,surprise}/raf_xxx.png
│   └── 1040x1568/...
└── final_data_v1_bucket_postprocessed/     # KDEF / Multi-PIE / Oulu 等合集
    └── 544x736/{neutral,angry,...}/{kdef|multi_pie|oulu}_xxx.JPG
```

本仓库的 `configs/stage1.yaml` 与 `configs/stage2.yaml` 已默认指向上述相对路径。如果直接使用预处理数据，通常只需下载数据后开始训练；但正式训练时更推荐将 `data.sources[*].jsonl`、`data.sources[*].src_root`、`data.sources[*].params_root` 统一改成绝对路径，避免从不同工作目录启动脚本时出现 `os.path.relpath` 前缀不一致的问题。例如：

```yaml
data:
  sources:
    - jsonl: /abs/path/to/ControlFace-main/face_emoji/v1_pairs_with_instructions.jsonl
      src_root: /abs/path/to/ControlFace-main/face_emoji/final_data_v1_bucket_postprocessed
      params_root: /abs/path/to/ControlFace-main/face_emoji/deca_params/v1
    - jsonl: /abs/path/to/ControlFace-main/face_emoji/raf_pairs_with_instructions.jsonl
      src_root: /abs/path/to/ControlFace-main/face_emoji/final_data_raf_bucket_postprocessed
      params_root: /abs/path/to/ControlFace-main/face_emoji/deca_params/raf
```

注意：`jsonl` 中的 `image_a_path` / `image_b_path` 与 yaml 中的 `src_root` 必须使用同一种根目录写法。若你把 yaml 改成绝对路径，也建议同步将 jsonl 中的图像路径改成绝对路径；否则训练侧根据 `image_path -> params_root` 映射 `.pt` 时可能找不到对应的 DECA 参数。

命名约定（决定后续 `extract_person_id` 行为）：

| 前缀 | 子集 | person_id 提取规则 |
|---|---|---|
| `raf_*` | RAF-DB | 文件名去扩展名整体作 person_id |
| `kdef_*` | KDEF | 取文件名中第 6-8 位 |
| `oulu_*` | Oulu-CASIA | 取下划线分割的第 1 段 |
| `multi_pie_*` | Multi-PIE | 取下划线分割的第 2 段 |

如果你使用自建数据集，只需按上面格式组织：`<root>/<bucket>/<expression>/<prefix>_<id>.<ext>`，并在 `scripts/generate_pairs_jsonl.py:extract_person_id` 中扩展前缀解析，然后继续执行下面的数据准备流程。

### 2.2 Generate Expression-Pair JSONL

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
  "image_a_path": "./face_emoji/final_data_v1_bucket_postprocessed/544x736/surprise/kdef_AF02SUS.JPG", 
  "image_a_filename": "kdef_AF02SUS.JPG", 
  "expression_a": "surprise", 
  "image_b_path": "./face_emoji/final_data_v1_bucket_postprocessed/544x736/neutral/kdef_AF02NES.JPG", 
  "image_b_filename": "kdef_AF02NES.JPG", 
  "expression_b": "neutral",
  "check_result": null
}
```

> 如果你直接使用 Hugging Face 上的 `*_pairs_with_instructions.jsonl`，可以跳过 2.2 和 2.3；只有在准备自己的数据集或重新生成配对时才需要执行下面两步。

### 2.3 LLM Quality Filtering and Instruction Generation

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
  "image_a_path": "./face_emoji/final_data_v1_bucket_postprocessed/544x736/surprise/kdef_AF02SUS.JPG", 
  "image_a_filename": "kdef_AF02SUS.JPG", 
  "expression_a": "surprise", 
  "image_b_path": "./face_emoji/final_data_v1_bucket_postprocessed/544x736/neutral/kdef_AF02NES.JPG", 
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

### 2.4 Offline DECA Parameter Extraction

训练时不会在线跑 DECA encode（太慢），而是预先把每张图的 DECA 编码结果离线保存为 `.pt`：

**单卡：**
```bash
python scripts/extract_deca_params.py \
    --src_root ./face_emoji/final_data_raf_bucket_postprocessed \
    --out_root ./face_emoji/deca_params/raf \
    --src_root ./face_emoji/final_data_v1_bucket_postprocessed \
    --out_root ./face_emoji/deca_params/v1 \
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
./face_emoji/deca_params/raf/544x736/angry/raf_xxx.pt   # dict(shape, tex, exp, pose, cam, light, detail, tform)
```

断点续跑：脚本会自动跳过已存在的 `.pt`，多卡互相不重叠。失败样本写入 `{out_root}/_failed_shard{i}.jsonl`。

### 2.5 Verify Extracted Parameters

```bash
python scripts/verify_deca_params.py \
    --src_root ./face_emoji/final_data_raf_bucket_postprocessed \
    --out_root ./face_emoji/deca_params/raf \
    --src_root ./face_emoji/final_data_v1_bucket_postprocessed \
    --out_root ./face_emoji/deca_params/v1 \
    --deep_check
```

会汇总：
- `_missing.txt`：源图存在但 `.pt` 缺失（需重跑 2.4）
- `_orphan.txt`：多余的 `.pt`（源图已删，可清理）
- `_dup_fail.txt`：既写入又出现在 failed 列表的异常样本
- `--deep_check` 还会逐个 `torch.load` 验证每个字段的 shape

---

## 3. Stage-1 Training (Conditional DECA Encoder)

<p align="center">
  <img src="docs/figures/stage1_training.png" width="860" alt="Stage1 training architecture">
</p>

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
```

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

## 4. Stage-2 Training (FLUX.2 Control Mixer)

<p align="center">
  <img src="docs/figures/overview.png" width="860" alt="Stage2 training architecture">
</p>

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
```


关键配置（详见 yaml）：

| 字段 | 默认 | 说明 |
|---|---|---|
| `stage1.ckpt_path` | `./checkpoints/stage1/.../best-step-{N}.pt` | **必须**指向上一步训练好的 Stage1 ckpt |
| `stage2.resume_path` | null | 若中断，可填某 ckpt 续训 |
| `stage2.detach_deca_control` | true | true 时 flow loss 不反传 DECA renderer，省显存；false 走完整端到端 |
| `flux.model_path` | FLUX.2-klein-base 目录 | 需 Diffusers 格式；klein-base **非蒸馏**，transformer guidance 必须传 None |
| `model.control_mixer_*` | 512 / 8 / 128 | CMM 结构（推理 yaml 必须与训练保持一致） |
| `data.use_alpha_mask` | true | 控制图是否用 alpha mask 把背景置 0 |
| `loss.lambda_flow / lambda_aux` | 1.0 / 0.1 | flow loss 与 Stage1 aux loss 的权重 |
| `train.gradient_checkpointing` | true | 推荐开启，FLUX.2 单卡 80G 才放得下 batch=1 |

**输出：** `./checkpoints/stage2/stage2-<timestamp>/best-step-{N}.pt`，包含 `{stage1_model, control_mixer, cfg, step, ...}`，是推理时唯一需要的权重。

---

## 5. Inference

<p align="center">
  <img src="docs/figures/rcg_inference.svg" width="900" alt="Reference Control Guidance inference workflow">
</p>

### 5.1 RCG (Reference Control Guidance)

推理阶段先构造两组控制条件：参考控制 `D_R` 由参考图完整 DECA 参数渲染得到，目标控制 `D_T` 只替换 Stage1 预测的目标表情和下颌参数，其余身份、纹理、相机和光照参数保持参考图不变。随后 CMM 进行两次前向：第一次输入 `(D_R, D_R)` 得到参考噪声预测，第二次输入 `(D_T, D_R)` 得到目标噪声预测，并沿目标-参考差异方向外推：

```
eps_ref = DiT(cat(latents, ref_latents, CMM(D_R, D_R)), text)
eps_tgt = DiT(cat(latents, ref_latents, CMM(D_T, D_R)), text)
eps     = eps_ref + λ · (eps_tgt - eps_ref)
```

| λ | 行为 |
|---|---|
| 0.0 | 生成 ≈ 参考图（几乎不变） |
| 1.0 | 等价于无 RCG（eps = eps_tgt） |
| 3.0 | 默认，表情清晰且身份稳 |
| >3 | 表情更夸张，但身份漂移风险上升 |

RCG 以参考控制为基准，沿目标表情方向放大引导；`λ` 控制表情变化幅度，通常可在 `0/1/3/5/7` 中 sweep 选择。

### 5.2 Stage-1 Inference (Control Map Visualization)

用于验证 Stage1 是否能根据 prompt 输出合理的表情几何，输出 6 张图（参考路径 D_R + 目标路径 D_T，各 rendered/normal/albedo）+ 2 个 9ch tensor。

```bash
python infer/infer_stage1.py \
    --ref     ./原图.png \
    --prompt  "make her burst into laughter" \
    --output_dir ./output_stage1 \
    --ckpt    ./checkpoints/stage1/stage1-<timestamp>/best-step-{N}.pt
```

### 5.3 Stage-2 Inference (End-to-End Expression Editing)

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
    --stage2_ckpt ./checkpoints/stage2/.../best-step-{N}.pt \
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
    --stage2_ckpt ./checkpoints/stage2/.../best-step-{N}.pt \
    --rcg_enabled false \
    --output_dir ./output_stage2/no_rcg
```

**临时覆盖 yaml 任意字段：**
```bash
python infer/infer_stage2.py --config configs/infer_stage2.yaml \
    --ref 原图.png --prompt "smile" \
    --stage2_ckpt ./checkpoints/stage2/.../best-step-{N}.pt \
    --opts sampling.num_inference_steps=28 sampling.height=512 sampling.width=512 \
           sampling.rcg.lambda=2.5
```

推理输出（默认 `output_stage2/`）：
- `final.png`：FLUX.2 生成的最终图
- `D_R_*.png` / `D_T_*.png`：参考/目标控制图
- `*_9ch.pt` / `*_tokens.pt`：调试中间张量（`output.save_intermediates=true` 才生成）
- `summary.json`：本次推理的所有有效配置

---

## 6. Configuration Files

本仓库共有三个 yaml 配置文件，分别对应训练 / 推理的不同阶段，所有可调字段都已在文件中带详细中文注释，按需修改即可：

| 配置文件 | 作用 |
|---|---|
| [`configs/stage1.yaml`](configs/stage1.yaml) | Stage1 训练：Conditional DECA Encoder 的数据源 / 模型结构 / 优化器 / loss / ckpt 策略 |
| [`configs/stage2.yaml`](configs/stage2.yaml) | Stage2 训练：FLUX.2 Control Mixer 的端到端联合训练，引用 Stage1 ckpt + FLUX.2 主干 + CMM 结构 |
| [`configs/infer_stage2.yaml`](configs/infer_stage2.yaml) | Stage2 推理：FLUX.2 采样参数 + RCG 系数 + Stage2 ckpt 路径，结构字段须与 stage2.yaml 保持一致 |

所有字段也支持通过 `--opts key=val` 在 CLI 临时覆盖，无需修改文件。

---

## 7. Repository Structure

```
ControlFace-main/
├── configs/                       # 所有 yaml 配置
│   ├── stage1.yaml                # Stage1 训练配置
│   ├── stage2.yaml                # Stage2 训练配置
│   └── infer_stage2.yaml          # Stage2 推理配置
├── data/                          # DECA 静态资产 (head_template.obj / mask 等)
├── decalib/                       # DECA 官方代码 (encoder / FLAME / renderer)
├── face_emoji/                    # Hugging Face 下载的数据目录 (默认不纳入 git)
│   ├── raf_pairs_with_instructions.jsonl
│   ├── v1_pairs_with_instructions.jsonl
│   ├── final_data_raf_bucket_postprocessed/
│   ├── final_data_v1_bucket_postprocessed/
│   └── deca_params/{raf,v1}/<bucket>/<expr>/*.pt
├── infer/
│   ├── infer_stage1.py            # 控制图可视化
│   └── infer_stage2.py            # 端到端表情编辑
├── prompts/check_og_data.txt      # 火山方舟质检 prompt 模板
├── scripts/
│   ├── generate_pairs_jsonl.py    # 步骤 2.2: 生成配对 jsonl
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
├── requirements.txt
├── set_env.sh                     # 一键安装
└── README.md
```

---

## Acknowledgements

本项目在以下工作的基础上构建：
- [ControlFace (CVPR 2025)](https://github.com/cvlab-kaist/ControlFace) – 控制范式与整体架构
- [DECA](https://github.com/yfeng95/DECA) / [DiffusionRig](https://github.com/adobe-research/diffusion-rig) – 3D 人脸几何编码与渲染
- [FLUX.2-klein-base](https://huggingface.co/black-forest-labs/FLUX.2-klein-base) – 主干流匹配模型
- [face-alignment (FAN)](https://github.com/1adrianb/face-alignment) – 在线人脸关键点检测

感谢上游作者们开源的工作。

---

<a id="en"></a>

## Abstract

This repository implements **3DMM-FLUX**, an instruction-guided facial expression editing framework built on parametric 3D face representation and FLUX diffusion/flow matching. Given a single face image and a natural-language instruction, the system first estimates identity-aware geometry, pose, expression, camera, and illumination parameters with DECA/3DMM. It then predicts the target expression parameters, renders expression-aware geometric control signals, and injects them into the FLUX.2-klein flow-matching generation process. The framework combines interpretable 3D facial priors with high-fidelity generative modeling, aiming to improve expression controllability while preserving the input identity.

The implementation follows the control paradigm of [ControlFace (CVPR 2025)](https://github.com/cvlab-kaist/ControlFace), uses [DECA](https://github.com/yfeng95/DECA) for 3DMM parameter estimation and rendering, and adopts the official [FLUX.2-klein-base](https://huggingface.co/black-forest-labs/FLUX.2-klein-base) configuration, including its paired text encoder. Training is performed on the in-house FacePairEmoji expression-pair dataset with both Chinese and English editing instructions.

## Highlights

- **Parametric expression control**: DECA/3DMM decomposes an input face into identity, pose, expression, illumination, and camera parameters, providing explicit geometric supervision.
- **Instruction-driven editing**: Chinese and English prompts are encoded with the FLUX.2-klein text encoder and used to predict target DECA expression and jaw-pose parameters.
- **FLUX.2 flow-matching generation**: Reference and target 3DMM control maps are projected into control tokens and injected into the FLUX.2-klein transformer.
- **Identity-preserving inference**: Reference-Control Guidance (RCG) strengthens identity consistency and enables direct comparison with the vanilla FLUX.2 baseline.

## Demo

The qualitative matrix is shared with the Chinese section to avoid duplicating a large set of images in the README. The rows and columns follow the same expression order: `Neutral -> Angry -> Disgust -> Fear -> Happy -> Sad -> Surprise`. Each row corresponds to one input identity, and the left column only reports the original image resolution. The pale-yellow diagonal cells show the input images, while all off-diagonal cells directly reference the original generated results under `demo_output/`. Each off-diagonal cell contains a nested 2×2 comparison: the top row is our controlled model (`Control-CN / Control-EN`), and the bottom row is the vanilla FLUX.2 baseline without 3DMM control (`OG-CN / OG-EN`).

View the matrix here: [Demo](#demo).

## 1. Environment

We recommend the `controlface310` conda environment with CUDA 12.1 and PyTorch 2.5.1:

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

The same setup commands are collected in `set_env.sh`:

```bash
bash set_env.sh
```

### PyTorch3D Installation

The DECA renderer uses `pytorch3d`. For PyTorch 2.5, installing PyTorch3D from source is recommended:

```bash
git clone https://github.com/facebookresearch/pytorch3d.git
cd pytorch3d
pip install --no-build-isolation -e .
```

### DECA Setup

Required pretrained assets:

| Asset | Target path | Note |
|---|---|---|
| `deca_model.tar` | `data/deca_model.tar` | DECA pretrained checkpoint |
| `generic_model.pkl` | `data/generic_model.pkl` | FLAME 2020 model |
| `FLAME_texture.npz` | `data/FLAME_texture.npz` | FLAME texture space |
| `head_template.obj`, `uv_face_eye_mask.png`, `fixed_displacement_256.npy`, `mean_texture.jpg` | `data/` | DECA static assets |

### FLUX.2-klein-base

Download `black-forest-labs/FLUX.2-klein-base-4B` in Diffusers format and point the corresponding yaml fields to the local model directory.

## 2. Data Preparation

### 2.1 Download Preprocessed Dataset (FacePairEmoji)

The released FacePairEmoji dataset is available at [`yunpengZhangup/FacePairEmoji`](https://huggingface.co/datasets/yunpengZhangup/FacePairEmoji). It contains the processed image folders, instruction jsonl files, and pre-extracted DECA parameters, so you can train directly without regenerating the metadata or DECA features:

```bash
hf download yunpengZhangup/FacePairEmoji --repo-type=dataset \
  --local-dir ./face_emoji
```

The downloaded directory should follow this layout:

```text
face_emoji/
  raf_pairs_with_instructions.jsonl
  v1_pairs_with_instructions.jsonl
  deca_params/
    raf/
    v1/
  final_data_raf_bucket_postprocessed/
  final_data_v1_bucket_postprocessed/
```

The default `configs/stage1.yaml` and `configs/stage2.yaml` point to this `./face_emoji/...` layout. For full training runs, absolute paths are recommended for `data.sources[*].jsonl`, `data.sources[*].src_root`, and `data.sources[*].params_root`, especially when launching jobs from different working directories:

```yaml
data:
  sources:
    - jsonl: /abs/path/to/ControlFace-main/face_emoji/v1_pairs_with_instructions.jsonl
      src_root: /abs/path/to/ControlFace-main/face_emoji/final_data_v1_bucket_postprocessed
      params_root: /abs/path/to/ControlFace-main/face_emoji/deca_params/v1
    - jsonl: /abs/path/to/ControlFace-main/face_emoji/raf_pairs_with_instructions.jsonl
      src_root: /abs/path/to/ControlFace-main/face_emoji/final_data_raf_bucket_postprocessed
      params_root: /abs/path/to/ControlFace-main/face_emoji/deca_params/raf
```

The image paths stored in the jsonl files and the `src_root` values in the yaml files must use the same root convention. If you convert yaml paths to absolute paths, it is safer to convert `image_a_path` and `image_b_path` in the jsonl files to absolute paths as well; otherwise the dataset loader may fail to map an image path to the corresponding `.pt` file under `params_root`.

### 2.2 Generate Expression-Pair JSONL

If you want to use your own dataset, organize it as `<root>/<bucket>/<expression>/<prefix>_<id>.<ext>`, then generate expression-pair metadata:

```bash
python scripts/generate_pairs_jsonl.py \
  --data_dir ./face_emoji/final_data_raf_bucket_postprocessed \
  --output ./raf_pairs.jsonl

python scripts/generate_pairs_jsonl.py \
  --data_dir ./face_emoji/final_data_v1_bucket_postprocessed \
  --output ./v1_pairs.jsonl
```

### 2.3 LLM Quality Filtering and Instruction Generation

The quality-filtering and instruction-generation stage is released as prompt templates under `prompts/`. API credentials and vendor-specific request code are intentionally not stored in this repository; connect the templates to your preferred multimodal or text LLM provider when reproducing this step.

### 2.4 Offline DECA Parameter Extraction

If you do not use the released pre-extracted DECA parameters, extract them offline before training:

```bash
bash scripts/run_extract_multigpu.sh
```

### 2.5 Verify Extracted Parameters

Verify the extracted parameters:

```bash
python scripts/verify_deca_params.py \
  --src_root ./face_emoji/final_data_raf_bucket_postprocessed \
  --out_root ./face_emoji/deca_params/raf \
  --src_root ./face_emoji/final_data_v1_bucket_postprocessed \
  --out_root ./face_emoji/deca_params/v1 \
  --deep_check
```

## 3. Stage-1 Training

<p align="center">
  <img src="docs/figures/stage1_training.png" width="860" alt="Stage1 training architecture">
</p>

Stage 1 trains a text-conditioned DECA expression encoder:

- Input: reference face image and text instruction.
- Output: target expression coefficients `exp[:50]` and jaw pose `pose[3:6]`.
- Supervision: offline DECA parameters extracted from the target image.
- Text encoder: the official FLUX.2-klein Qwen3 encoder, concatenating hidden states from layers 9, 18, and 27.

Configuration: [`configs/stage1.yaml`](configs/stage1.yaml)

```bash
bash train/train_stage1.sh
```

The best checkpoint is saved under `checkpoints/stage1/stage1-<timestamp>/best-step-{N}.pt`.

## 4. Stage-2 Training

<p align="center">
  <img src="docs/figures/overview.png" width="860" alt="Stage2 training architecture">
</p>

Stage 2 jointly trains the FLUX.2 control pathway:

1. Stage 1 predicts target DECA expression and jaw parameters.
2. DECA renders reference and target control maps, including rendered image, normal, and albedo channels.
3. `Flux2ControlMixer` projects the control maps into tokens and injects them into the FLUX.2 transformer.
4. Flow-matching loss supervises generation in the FLUX.2 latent space, with an auxiliary Stage-1 loss for expression consistency.

Configuration: [`configs/stage2.yaml`](configs/stage2.yaml)

```bash
bash train/train_stage2.sh
```

The resulting checkpoint is saved under `checkpoints/stage2/stage2-<timestamp>/best-step-{N}.pt` and contains `stage1_model`, `control_mixer`, configuration snapshots, and training step metadata.

## 5. Inference

<p align="center">
  <img src="docs/figures/rcg_inference.svg" width="900" alt="Reference Control Guidance inference workflow">
</p>

### 5.1 Reference Control Guidance (RCG)

At inference time, the system renders two control conditions. The reference control `D_R` is rendered from the complete DECA parameters of the reference image, while the target control `D_T` only replaces expression and jaw parameters predicted by Stage 1; identity, texture, camera, and lighting remain inherited from the reference image. The Control Mixer then performs two forward passes:

```text
eps_ref = DiT(cat(latents, ref_latents, CMM(D_R, D_R)), text)
eps_tgt = DiT(cat(latents, ref_latents, CMM(D_T, D_R)), text)
eps     = eps_ref + lambda * (eps_tgt - eps_ref)
```

| lambda | Behavior |
|---|---|
| 0.0 | nearly unchanged reference image |
| 1.0 | equivalent to the target branch without RCG |
| 3.0 | default, clear expression editing with stable identity |
| >3 | stronger expression, with higher identity-drift risk |

RCG uses reference control as the identity anchor and amplifies the target-reference difference direction. The coefficient `lambda` controls the expression strength.

### 5.2 Stage-1 Inference

Stage-1 visualization renders the predicted 3DMM control maps:

```bash
python infer/infer_stage1.py \
  --config configs/stage1.yaml \
  --ckpt ./checkpoints/stage1/stage1-<timestamp>/best-step-{N}.pt \
  --image ./demo_input/256*256_surprise/raf_train_10109.png \
  --prompt "make the person look happy" \
  --out_dir ./output_stage1_demo
```

### 5.3 Stage-2 Inference

End-to-end FLUX.2 expression editing:

```bash
python infer/infer_stage2.py \
  --config configs/infer_stage2.yaml \
  --image ./demo_input/256*256_surprise/raf_train_10109.png \
  --prompt "make the person look happy" \
  --out_dir ./output_stage2_demo
```

The default output directory contains `final.png`, rendered reference/target control maps, optional intermediate tensors, and `summary.json`.

## 6. Configuration Files

| File | Purpose |
|---|---|
| [`configs/stage1.yaml`](configs/stage1.yaml) | Stage-1 data sources, model paths, optimization, losses, and checkpoint policy |
| [`configs/stage2.yaml`](configs/stage2.yaml) | Stage-2 FLUX.2 control-mixer training and Stage-1 checkpoint binding |
| [`configs/infer_stage2.yaml`](configs/infer_stage2.yaml) | End-to-end inference, FLUX.2 sampling options, RCG coefficient, and output settings |

All fields can also be overridden from the command line with `--opts key=val`.

## 7. Repository Structure

```text
ControlFace-main/
├── configs/              # YAML configs for training and inference
├── data/                 # DECA static assets
├── decalib/              # DECA encoder, FLAME, and renderer code
├── infer/                # Stage-1 and Stage-2 inference scripts
├── prompts/              # Prompt templates
├── scripts/              # Data preparation and DECA parameter extraction utilities
├── src/                  # Datasets, losses, and model modules
├── train/                # Stage-1 and Stage-2 training entry points
├── demo_input/           # Example input images used by the README gallery
├── demo_output/          # Example generated results used by the README gallery
├── requirements.txt
├── set_env.sh
└── README.md
```

## Acknowledgements

This project builds on the following open-source works:

- [ControlFace (CVPR 2025)](https://github.com/cvlab-kaist/ControlFace)
- [DECA](https://github.com/yfeng95/DECA) and [DiffusionRig](https://github.com/adobe-research/diffusion-rig)
- [FLUX.2-klein-base](https://huggingface.co/black-forest-labs/FLUX.2-klein-base)
- [face-alignment (FAN)](https://github.com/1adrianb/face-alignment)
