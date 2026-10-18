# BiomedCoOp 阅读笔记

[论文原文](../papers/biomedcoop.pdf) · [返回论文列表](../README.md#2-论文列表与阅读记录) · [主实验复现报告](../reproduction/biomedcoop-biomedclip/复现报告.md)

- **论文**：BiomedCoOp: Learning to Prompt for Biomedical Vision-Language Models
- **会议**：CVPR 2025
- **链接**：[论文主页](https://openaccess.thecvf.com/content/CVPR2025/html/Koleilat_BiomedCoOp_Learning_to_Prompt_for_Biomedical_Vision-Language_Models_CVPR_2025_paper.html)｜[官方代码](https://github.com/HealthX-Lab/BiomedCoOp)

## 一句话理解

冻结 BiomedCLIP，用少量标注学习文本 prompt，并通过 LLM 生成的医学描述约束学习方向，减少过拟合，提高对微调时未见类别的识别能力。

## 研究问题

医学图像标注少，手写 prompt 又依赖专业知识。CoOp 虽然可以自动学习 prompt，但容易过度适应训练类别，损害新类别上的表现。

本文希望同时实现：**少样本适应当前任务，并保留预训练模型的类别迁移能力。**

## 方法

1. **生成描述**：GPT-4 根据类别名称和成像模态，为每个类别生成 50 条医学描述，用冻结的文本编码器提取特征。
2. **学习 prompt**：默认学习 4 个各类别共享的 context token，拼接类别名称；图像和文本编码器权重冻结。
3. **SCCM：约束文本表示**。将全部描述特征取平均，让 learned prompt 的特征靠近这个平均表示。
4. **KDSP：约束预测行为**。根据训练 batch 的图文匹配分数筛除离群描述，以剩余描述构造教师，通过 KL 散度指导学生的分类概率。

总损失：

$$
\mathcal L=\mathcal L_{\mathrm{CE}}+\lambda_1\mathcal L_{\mathrm{SCCM}}+\lambda_2\mathcal L_{\mathrm{KDSP}}.
$$

**关键细节**：SCCM 保留全部描述，KDSP 才筛选描述。前者保持语义覆盖，后者提供与图像相关的预测指导。推理时使用 learned prompt，无需调用 GPT-4。

## 主要实验结果

Few-shot 实验覆盖 11 个数据集、9 种成像模态。下表为跨数据集平均准确率（%）。

| 方法 | 1-shot | 16-shot |
| --- | ---: | ---: |
| CoOp | 50.16 | 69.62 |
| Tip-Adapter-F | 51.17 | 70.91 |
| XCoOp（补充实验） | 52.50 | 71.04 |
| **BiomedCoOp** | **57.03** | **72.42** |

- **类别迁移**：10 个数据集上的 Novel 类别平均准确率为 73.92%，CoOp 为 64.75%。这里的 Novel 指未参与此次微调的类别，推理仍提供类别名称。
- **描述本身很重要**：零样本 BiomedCLIP 从单模板的 42.05% 提升到 LLM 描述平均的 52.27%，说明部分收益来自更丰富的文本先验。

这篇论文值得记住的是：**少样本 prompt learning 需要合适的知识约束，教师描述的质量与筛选方式会影响迁移效果。**

