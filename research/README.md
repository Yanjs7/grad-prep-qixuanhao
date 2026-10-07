# 科研任务总览

[返回仓库首页](../readme.md)

本目录收录医学图像分析相关论文与个人阅读记录，当前收录 10 篇论文的阅读笔记，10 篇均已附论文 PDF。

## 当前进度与实践入口

| 内容 | 当前状态 | 入口 |
| --- | --- | --- |
| 论文阅读 | 10 篇已有笔记，继续深化精读 | 见下方论文列表 |
| U-Net 血管分割实践 | 已有源码压缩包、实验报告与可视化 | [FIVES 报告](<../engineering/U-Net 在 FIVES 上的血管分割/U-net复现.md>) |
| U-Net 病灶分割与 ResNet18 分类 | 已有验证指标、训练曲线与误判分析 | [ISIC 实验报告](../engineering/基于U-net和ResNet的皮肤病灶分析与分类平台/项目实验报告.md) |

## 阅读内容与研究思考

### 1. 学习主线

目前的论文阅读与实践围绕以下几个问题展开：

1. **如何完成像素级分割？** 从 U-Net 的编码器、解码器和跳跃连接出发，结合眼底血管与皮肤病灶实验理解训练过程。
2. **如何建立可靠的实验流程？** 通过 nnU-Net 学习预处理、模型配置与训练策略之间的关系，关注数据划分、模型选择和评价口径。
3. **如何理解注意力与提示式分割？** 阅读 Transformer、SAM 与 SAM 2，梳理基础结构、交互方式和图像／视频任务之间的联系。
4. **如何从模型走向实际应用？** 结合 iAorta 的阅读笔记与自己的平台实践，思考定位、分割、分类和结果展示如何服务具体问题。
5. **如何减少分割标注需求？** 结合 DTC、BCP、DeSCO 与 Swin UNETR，区分训练阶段利用未标注数据、减少每份影像的标注切片，以及先自监督预训练再微调这几种路径。

### 2. 论文列表与阅读记录

| 论文名称 | 学习重点 | 论文原文 | 阅读笔记 | 实践情况 |
| --- | --- | --- | --- | --- |
| U-Net: Convolutional Networks for Biomedical Image Segmentation | 编码器—解码器结构、跳跃连接、像素级预测 | [PDF](papers/U-Net.pdf) | [U-Net 笔记](papernotes/unet.md) | 已开展 FIVES 与 ISIC 数据集上的模型实践 |
| nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation | 数据驱动的配置、预处理与训练流程 | [PDF](papers/nnU-Net.pdf) | [nnU-Net 笔记](papernotes/nnunet.md) | 已有阅读笔记，尚未整理独立复现 |
| Attention Is All You Need | 自注意力与 Transformer 基础结构 | [PDF](papers/Transformer.pdf) | [Transformer 笔记](papernotes/transformer.md) | 已有阅读笔记，尚未整理独立复现 |
| Segment Anything | 提示式分割、模型使用方式与数据构建 | [PDF](papers/SAM.pdf) | [SAM 笔记](papernotes/SAM.md) | 已有阅读笔记，尚未整理独立复现 |
| SAM 2: Segment Anything in Images and Videos | 图像与视频分割、与 SAM 的联系和区别 | [PDF](papers/SAM2.pdf) | [SAM 2 笔记](papernotes/SAM2.md) | 已有阅读笔记，尚未整理独立复现 |
| AI-based diagnosis of acute aortic syndrome from noncontrast CT | 两阶段医学影像分析、多任务学习与应用流程 | [PDF](papers/iAorta.pdf) | [iAorta 笔记](papernotes/iAorta.md) | 已有阅读笔记，尚未整理独立复现 |
| Semi-supervised Medical Image Segmentation through Dual-task Consistency | 双任务一致性与未标注影像的训练约束 | [PDF](papers/DTC.pdf) | [DTC 笔记](papernotes/dtc.md) | 已有阅读笔记，尚未整理独立复现 |
| Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation | 双向复制粘贴、Teacher–Student 与混合标签训练 | [PDF](papers/BCP.pdf) | [BCP 笔记](papernotes/bcp.md) | 已有阅读笔记，尚未整理独立复现 |
| Orthogonal Annotation Benefits Barely-Supervised Medical Image Segmentation | 正交切片标注、标签传播与三维分割 | [PDF](papers/DeSCO.pdf) | [DeSCO 笔记](papernotes/desco.md) | 已有阅读笔记，尚未整理独立复现 |
| Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis | 三维 CT 自监督预训练与有标注数据微调 | [PDF](papers/Swin.pdf) | [Swin UNETR 笔记](papernotes/swin-unetr.md) | 已有阅读笔记，尚未整理独立复现 |



### 3. 现阶段理解与待验证问题

- **分割质量与应用效果需要分别评估。** Dice、IoU 用于观察分割效果；平台还需要关注输入处理、模型加载、历史保存与结果导出的可靠性。
- **整体分类准确率不足以解释各类别表现。** 现有皮肤病灶实验中，黑色素瘤的验证召回率为 0.60，需要结合分类别指标与混淆矩阵分析误判。
- **复杂模型应与清晰的基线比较。** 后续引入新的结构或策略时，应尽量保持数据划分和评价口径一致，通过对照实验判断收益来源。

以下是准备进一步验证的候选问题，目前尚未形成研究结论：

| 候选问题 | 初步验证思路 | 关注指标 |
| --- | --- | --- |
| 病灶分割结果能否辅助分类？ | 固定数据划分，比较整图、预测掩码生成的 ROI 与保留周围上下文的 ROI；真实标注 ROI 仅作参考上限 | 宏平均 F1、分类别召回率、混淆矩阵 |
| 如何改善少数类别的识别？ | 固定划分与训练预算，对比加权交叉熵、采样或损失策略；只在验证集调参，最终统一评估测试集 | 黑色素瘤召回率、Precision、宏平均 F1 |
| 分割模型在哪些样本上容易失败？ | 按病灶大小、边界清晰程度和图像干扰整理失败样例 | Dice、IoU、预测与标注的可视化对照 |

上述问题属于待验证的实验方向，创新性需通过文献对照确认。实验应避免使用测试集选择模型；ROI 分类的验证／测试输入应由预测掩码生成，以贴近实际使用流程。后续可在独立研究想法文档中补充问题动机、相关工作、实验方案和候选投稿会议／期刊。

## 少标签方法对照

| 论文 | 使用标注的方式 | 未标注数据的作用 | 阅读时需区分的概念 |
| --- | --- | --- | --- |
| DTC | 少量完整标注影像提供监督 | 分割与边界相关预测保持一致 | 一致性不能保证预测正确 |
| BCP | 完整真实标签与伪标签按区域混合 | Teacher 生成伪标签，双向复制粘贴用于训练 | 伪标签区域与真实标注区域的可信度不同 |
| DeSCO | 每份影像的两张正交切片 | 标签传播和模型间相互指导 | 少切片标注不同于少量完整标注病例 |
| Swin UNETR | 在微调阶段使用分割标注 | 在预训练阶段学习影像表征 | 自监督预训练与半监督分割的训练流程不同 |

