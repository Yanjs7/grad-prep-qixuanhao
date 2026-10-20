# grad-prep-qixuanhao

本仓库记录我的论文阅读、深度学习实验和工程实践。

## 个人简介

- **姓名**：戚轩豪
- **专业**：计算机科学与技术
- **当前学习主题**：深度学习、医学图像分析、图像分割与分类、医疗多模态表示学习。
- **已开展的实践**：FIVES 眼底血管分割、U-Net 与 ResNet18 皮肤病灶平台，以及 主动脉夹层CTA半监督研究。
- **学习目标**：在掌握基础模型的同时，提高论文分析、实验设计、代码实现与结果解释能力，为后续科研与论文写作积累基础。

## 内容导航

- [任务进度总览](#任务进度总览)
- [已完成内容速览](#已完成内容速览)
- [科研学习](#科研学习)
- [工程实践](#工程实践)

## 任务进度总览

### 科研任务

| 项目 | 状态 | 当前进度 |
| --- | --- | --- |
| 论文阅读与笔记 | 持续完善 | 13 篇已有笔记，继续补充方法、实验与局限分析 |
| 模型实现与实验 | 已有成果 | 3项工程实践 + 1项论文主实验复现：FIVES、ISIC、CTA及BiomedCoOp / BiomedCLIP |
| 论文复现 | 已完成BiomedCoOp主实验 | 11数据集、5个shot档位、3种子，506项配置；[图文报告](research/reproduction/biomedcoop-biomedclip/复现报告.md) |

科研总览：[research/README.md](research/README.md)。

### 工程任务

| 项目 | 状态 | 已有交付物 |
| --- | --- | --- |
| U-Net 在 FIVES 上的眼底血管分割 | 已有实验成果 | 源码压缩包、实验报告、训练曲线与预测样例 |
| U-Net + ResNet18 皮肤病灶平台 | 已有平台与验证结果 | 使用文档、源码压缩包、实验报告、界面与训练曲线 |
| 主动脉夹层CTA半监督研究 | 完成固定五折实验 | 详细报告、数据集与案例配图、指标与折外预测、训练日志、冻结代码和协议 |

工程总览：[engineering/README.md](engineering/README.md)。

## 已完成内容速览

### 论文笔记

- [U-Net](research/papernotes/unet.md)：记录 U 形结构、跳跃连接和分割模型的学习过程，并结合两个数据集开展实践。
- [nnU-Net](research/papernotes/nnunet.md)：关注固定参数、规则参数和经验参数，以及如何组织可复用的实验流程。
- [Transformer](research/papernotes/transformer.md)、[SAM](research/papernotes/SAM.md) 与 [SAM 2](research/papernotes/SAM2.md)：梳理注意力基础、提示式分割及图像／视频分割的联系。
- [iAorta](research/papernotes/iAorta.md)：理解主动脉定位、多任务分析与临床预警流程，记录对多模态扩展的思考。
- [DTC](research/papernotes/dtc.md)：双任务一致性与未标注影像的训练约束。
- [BCP](research/papernotes/bcp.md)：双向复制粘贴、Teacher–Student 与混合标签训练。
- [DeSCO](research/papernotes/desco.md)：正交切片标注、标签传播与三维分割。
- [Swin UNETR](research/papernotes/swin-unetr.md)：三维 CT 自监督预训练与有标注数据微调。
- [EHR多模态表示学习综述](research/papernotes/ehr-multimodal-survey.md)：梳理早期、中间、晚期与混合融合，以及模态缺失、数据对齐和跨医院差异。
- [BiomedCoOp](research/papernotes/biomedcoop.md)：理解冻结医学视觉语言模型后的少样本提示学习，以及语义约束、描述筛选与类别迁移。
- [Gastric-X](research/papernotes/gastric-x.md)：理解胃癌多模态、多期相数据与评测基准，区分检验文本、空间提示和不同任务的输入范围。

### 实验与应用

| 数据集 | 模型 / 任务 | 当前可查结果 | 结果入口 |
| --- | --- | --- | --- |
| FIVES | U-Net / 眼底血管分割 | 已有训练曲线和预测样例；报告尚待补充数值指标表 | [实验报告](<engineering/U-Net 在 FIVES 上的血管分割/U-net复现.md>) |
| ISIC 2017 | U-Net / 病灶分割 | 最佳验证 Dice 0.8476，同轮 IoU 0.7645 | [实验报告](engineering/基于U-net和ResNet的皮肤病灶分析与分类平台/项目实验报告.md) |
| ISIC 2017 | ResNet18 / 三类分类 | 最佳验证宏平均 F1 0.7907，同轮准确率 81.33% | [实验报告](engineering/基于U-net和ResNet的皮肤病灶分析与分类平台/项目实验报告.md) |
| 皮肤镜图像应用 | FastAPI Web 平台 | 图像上传、模型分析、历史查询、结果导出与打印 | [使用文档](engineering/基于U-net和ResNet的皮肤病灶分析与分类平台/readme.md) |
| 公开CTA（AortaSeg主分析） | MASS + MIL + Mean Teacher / 夹层类别判别 | S0+ / S1 OOF AUROC 0.7316 / 0.7263；主差异区间跨0 | [第三项实践报告](engineering/主动脉夹层CTA半监督研究/项目实验报告.md) |
| 11个医学分类数据集 | BiomedCLIP骨干 / BiomedCoOp少样本提示学习 | 506项配置完成；16-shot等权平均准确率72.55%，CoOp为70.91% | [复现报告](research/reproduction/biomedcoop-biomedclip/复现报告.md) |

## 科研学习

### 1. 学习主线

目前的论文阅读与实践围绕以下几个问题展开：

1. **如何完成像素级分割？** 从 U-Net 的编码器、解码器和跳跃连接出发，结合眼底血管与皮肤病灶实验理解训练过程。
2. **如何建立可靠的实验流程？** 通过 nnU-Net 学习预处理、模型配置与训练策略之间的关系，关注数据划分、模型选择和评价口径。
3. **如何理解注意力与提示式分割？** 阅读 Transformer、SAM 与 SAM 2，梳理基础结构、交互方式和图像／视频任务之间的联系。
4. **如何从模型走向实际应用？** 结合 iAorta 的阅读笔记与自己的平台实践，思考定位、分割、分类和结果展示如何服务具体问题。
5. **如何减少分割标注需求？** 结合 DTC、BCP、DeSCO 与 Swin UNETR，区分训练阶段利用未标注数据、减少每份影像的标注切片，以及先自监督预训练再微调这几种路径。
6. **如何构建医疗多模态患者表示？** 阅读EHR多模态表示学习综述，梳理医学图像、病历文字、检验数据与生理信号的融合方式，并关注模态缺失、时间对齐和跨医院差异。
7. **如何用少量标注适配医学视觉语言模型？** 结合BiomedCoOp阅读笔记，理解可学习文本提示、语义一致性与知识蒸馏，并区分训练类别上的适应和新类别上的迁移。

### 2. 论文列表与阅读记录

#### 基础模型与医学影像应用

| 论文名称 | 学习重点 | 论文原文 | 阅读笔记 |
| --- | --- | --- | --- |
| U-Net: Convolutional Networks for Biomedical Image Segmentation | 编码器—解码器结构、跳跃连接、像素级预测 | [PDF](research/papers/U-Net.pdf) | [U-Net 笔记](research/papernotes/unet.md) |
| nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation | 数据驱动的配置、预处理与训练流程 | [PDF](research/papers/nnU-Net.pdf) | [nnU-Net 笔记](research/papernotes/nnunet.md) |
| Attention Is All You Need | 自注意力与 Transformer 基础结构 | [PDF](research/papers/Transformer.pdf) | [Transformer 笔记](research/papernotes/transformer.md) |
| Segment Anything | 提示式分割、模型使用方式与数据构建 | [PDF](research/papers/SAM.pdf) | [SAM 笔记](research/papernotes/SAM.md) |
| SAM 2: Segment Anything in Images and Videos | 图像与视频分割、与 SAM 的联系和区别 | [PDF](research/papers/SAM2.pdf) | [SAM 2 笔记](research/papernotes/SAM2.md) |
| AI-based diagnosis of acute aortic syndrome from noncontrast CT | 两阶段医学影像分析、多任务学习与应用流程 | [PDF](research/papers/iAorta.pdf) | [iAorta 笔记](research/papernotes/iAorta.md) |

#### 少标签医学图像分割

| 论文名称 | 学习重点 | 论文原文 | 阅读笔记 |
| --- | --- | --- | --- |
| Semi-supervised Medical Image Segmentation through Dual-task Consistency | 双任务一致性与未标注影像的训练约束 | [PDF](research/papers/DTC.pdf) | [DTC 笔记](research/papernotes/dtc.md) |
| Bidirectional Copy-Paste for Semi-Supervised Medical Image Segmentation | 双向复制粘贴、Teacher–Student 与混合标签训练 | [PDF](research/papers/BCP.pdf) | [BCP 笔记](research/papernotes/bcp.md) |
| Orthogonal Annotation Benefits Barely-Supervised Medical Image Segmentation | 正交切片标注、标签传播与三维分割 | [PDF](research/papers/DeSCO.pdf) | [DeSCO 笔记](research/papernotes/desco.md) |
| Self-Supervised Pre-Training of Swin Transformers for 3D Medical Image Analysis | 三维 CT 自监督预训练与有标注数据微调 | [PDF](research/papers/Swin.pdf) | [Swin UNETR 笔记](research/papernotes/swin-unetr.md) |

#### 医疗多模态表示学习

| 论文名称 | 学习重点 | 论文原文 | 阅读笔记 |
| --- | --- | --- | --- |
| A survey on electronic health record driven multimodal representation learning | EHR驱动的患者表示、跨模态融合、模态缺失与数据对齐 | [PDF](research/papers/ehr-multimodal-survey.pdf) | [EHR多模态综述笔记](research/papernotes/ehr-multimodal-survey.md) |
| BiomedCoOp: Learning to Prompt for Biomedical Vision-Language Models | 医学视觉语言模型的少样本提示学习、语义一致性与知识蒸馏 | [PDF](research/papers/biomedcoop.pdf) | [BiomedCoOp 笔记](research/papernotes/biomedcoop.md) |
| Gastric-X: A Multimodal Multi-Phase Benchmark Dataset for Advancing Vision-Language Models in Gastric Cancer Analysis | 多期CT、检验文本与空间提示的多模态数据集及评测基准 | [PDF](research/papers/gastric-x.pdf) | [Gastric-X 笔记](research/papernotes/gastric-x.md) |



### 3. 论文复现：BiomedCoOp / BiomedCLIP

已完成以冻结BiomedCLIP为骨干的BiomedCoOp主实验，覆盖11个数据集、1/2/4/8/16-shot和3个种子：BiomedCoOp与CoOp各165次训练、线性探针165次拟合、零样本11次评估，共506项配置。

![BiomedCoOp总体少样本结果](research/reproduction/biomedcoop-biomedclip/images/06-overall.png)

16-shot下，BiomedCoOp跨数据集等权平均准确率为72.55%，CoOp为70.91%，线性探针为69.04%；相对CoOp高1.64个百分点，分数据集为6升5降。该结果不是BiomedCLIP预训练复现，也不表示全部数据集均有提升。

- [图文复现报告与27幅图](research/reproduction/biomedcoop-biomedclip/复现报告.md)
- [复现入口、代码、配置与文件说明](research/reproduction/biomedcoop-biomedclip/README.md)

## 工程实践

### 1. U-Net 在 FIVES 上的眼底血管分割

**项目目标**：通过医学图像分割任务，理解从数据读取到训练、模型选择、测试评估和结果导出的完整流程。

| 项目 | 当前记录 |
| --- | --- |
| 数据集 | FIVES 眼底血管分割数据集 |
| 数据规模 | 数据集训练部分 600 张、测试部分 200 张；训练部分内的验证划分尚待在报告中说明 |
| 图像处理 | 原始尺寸 2048 × 2048，使用 0.25 的缩放比例，输入尺寸 512 × 512 |
| 模型与环境 | U-Net；AutoDL 云 GPU 环境 |
| 已记录内容 | 报告描述了数据预处理、训练、验证模型选择、测试评估与结果导出流程，并提供训练曲线及预测样例 |
| 待补充内容 | 最佳轮次、量化测试指标、训练／验证划分、运行命令及原始日志；当前仅凭报告无法核对测试结论 |

- [源码压缩包](<engineering/U-Net 在 FIVES 上的血管分割/UNet源码.zip>)
- [实验报告](<engineering/U-Net 在 FIVES 上的血管分割/U-net复现.md>)

**训练曲线**

![FIVES 血管分割训练曲线](engineering/screenshots/curves.png)

**分割结果示例**：从左到右为原图、真实标注、模型预测。

![FIVES 血管分割结果](engineering/screenshots/temp.png)

### 2. 基于 U-Net 与 ResNet18 的皮肤病灶分析与分类平台

**项目目标**：将病灶分割、三类分类与网页交互结合，实现从图像上传到结果查看、历史查询和导出的应用流程。分割与分类分别处理输入图像、独立推理；当前分类模型不使用分割掩码或病灶 ROI。

#### 技术方案与功能

| 层次 | 技术与实现 |
| --- | --- |
| 分割模型 | PyTorch U-Net，输出病灶二值掩码 |
| 分类模型 | ImageNet 预训练 ResNet18，区分黑素细胞痣、黑色素瘤、脂溢性角化病 |
| 后端 | FastAPI、PyTorch、Pillow、NumPy、SQLite |
| 前端 | HTML、CSS、JavaScript，由同一服务提供网页与接口 |
| 推理部署 | 本地 CPU 推理，默认 2 个线程 |

- 上传皮肤镜图像，查看原图、病灶掩码与叠加视图。
- 展示三类预测概率、分割面积占比、图像尺寸和推理耗时。
- 保存分析历史，支持文件名搜索、类别筛选和日期筛选。
- 下载包含原图、掩码、叠加图与 JSON 结果的 ZIP 文件。
- 查看记录详情，打印报告或通过浏览器另存为 PDF。

#### 已记录的实验结果

使用 ISIC 2017 官方训练／验证划分，实际训练样本为 1999 张，验证样本为 150 张。官方训练集为 2000 张；分类数据检查记录将 `ISIC_0012970` 标为 `ISIC_0012127` 的重复项并排除，实际使用 1999 张。该记录直接证明分类数据的排除情况。下列指标均来自现有实验报告中的**验证集结果**，不作为独立测试集结果或论文原始指标的复现声明。

| 任务 | 最佳模型选择依据 | 最佳轮次 | 同轮验证结果 |
| --- | --- | ---: | --- |
| U-Net 病灶分割 | 逐图平均 Dice 最大 | 38 | Dice **0.8476**；IoU **0.7645** |
| ResNet18 三类分类 | 宏平均 F1 最大 | 31 | 宏平均 F1 **0.7907**；准确率 **81.33%** |

其中，黑色素瘤验证集共有 30 张，正确识别 18 张，召回率为 **0.60**。后续将重点检查误判样本，并在实验中同时报告各类别表现。

#### 项目入口

- [使用文档](engineering/基于U-net和ResNet的皮肤病灶分析与分类平台/readme.md)
- [实验报告与详细指标](engineering/基于U-net和ResNet的皮肤病灶分析与分类平台/项目实验报告.md)
- [项目源码压缩包](engineering/基于U-net和ResNet的皮肤病灶分析与分类平台/基于U-net和ResNet的皮肤病灶分析与分类平台.zip)

源码压缩包的项目根目录为 `isic2017-platform/`。进入该目录，按使用文档建立 Python 环境并运行 `bash start_web.sh`，默认访问 `http://127.0.0.1:8765`。**当前压缩包不含训练权重**；完成模型分析还需配置 `artifacts/segmentation.pth` 和 `artifacts/classification.pth`，具体见使用文档。

![皮肤病灶分析与分类平台主界面](engineering/基于U-net和ResNet的皮肤病灶分析与分类平台/screenshots/analysis-desktop.png)

该平台用于皮肤镜图像算法展示与研究；现有结果不能证明临床可靠性，预测不能替代临床诊断。


### 3. 主动脉夹层CTA半监督研究

**项目目标**：在相同父模型、监督访问与续训预算下，检验诊断未知 CTA 的教师—学生一致性训练是否优于继续监督训练。

| 项目 | 结果与口径 |
| --- | --- |
| 数据 | 712份扫描：224来源阳性、19阴性候选、469诊断未知；扫描数不等于独立患者数 |
| 模型 | MASS预训练编码器、位置融合与注意力MIL、EMA教师 |
| 训练 | 固定5折，S0每折600步，S0+与S1各400步；15个训练及15个评估阶段完成 |
| 主分析 | AortaSeg 29份OOF扫描；S0 / S0+ / S1 AUROC为0.6737 / 0.7316 / 0.7263 |
| 主比较 | S1−S0+为−0.0053，条件配对95%区间[−0.0737, 0.0632]，尚未得到明确半监督改善证据 |

- [详细报告、真实数据图与案例分析](engineering/主动脉夹层CTA半监督研究/项目实验报告.md)
- [项目入口、冻结代码与结果导航](engineering/主动脉夹层CTA半监督研究/README.md)
- [原始结果表](engineering/主动脉夹层CTA半监督研究/results/metrics.csv)
- [数据集图像与四类案例分析](engineering/主动脉夹层CTA半监督研究/项目实验报告.md#74-具体案例从输入阈值到分类结果)

#### 数据组成与来源

![CTA数据集来源与标签组成](engineering/主动脉夹层CTA半监督研究/figures/07_dataset_distribution.png)

正式纳入712份扫描，其中来源阳性224份（31.46%）、阴性候选19份（2.67%）、诊断未知469份（65.87%）。这说明未标注诊断的扫描占多数，但可靠阴性很少；有标签扫描中的阳性比例为92.18%，因此需要同时查看敏感度、特异度和同来源结果。

| 数据来源 | 纳入数量与标签 | 实验中的作用 | 真实图像 |
| --- | --- | --- | --- |
| ImageTAAD | 114份，全部来源阳性 | 有标签训练与外层测试；单来源不能评价特异度 | [TAAD_1预览](engineering/主动脉夹层CTA半监督研究/figures/taad01_roi.png) |
| ImageTBAD | 100份，全部来源阳性 | 有标签训练与外层测试；单来源不能评价特异度 | [101预览](engineering/主动脉夹层CTA半监督研究/figures/tbad101_roi.png) |
| AortaSeg | 29份：10阳性、19阴性候选 | 同来源主评价；每折另用5份验证扫描选择模型与阈值 | [Dissec_06预览](engineering/主动脉夹层CTA半监督研究/figures/dissec06_roi.png) |
| DB-SNet | 469份，诊断未知 | S1中的一致性训练抽样池，不用于计算诊断准确率 | [未知扫描预览](engineering/主动脉夹层CTA半监督研究/figures/dbexample_roi.png) |

上表中的AortaSeg 29份通过五折产生折外预测；每一折的验证与测试分区互斥，验证5份不是29份之外另加的独立队列。

#### 真实CTA图像与模型输入

![AortaSeg Dissec_06真实CTA及预测主动脉ROI](engineering/主动脉夹层CTA半监督研究/figures/dissec06_roi.png)

**怎么看这张图：** 左侧是冠状面和矢状面，其他小图是多个轴位采样位置；黄色为自动预测的主动脉ROI，用于确定输入区域。它没有标注夹层病灶，也不能代表完整病灶覆盖。此例来源标签为阳性，但三个阶段均漏判，说明定位结果与分类结果需要分别评价。

![Old_03模型输入块的中心层](engineering/主动脉夹层CTA半监督研究/figures/old03_patch.png)

**模型实际看什么：** 这是Old_03的15个输入块中心层，其中12个为名义块、3个为补充块。每个三维块尺寸为32×96×96，图中只展示其中心一层；模型对整块编码，再用注意力MIL聚合扫描分数。Old_03属于来源支持的阴性候选，后续分类变化见下表。

这些图像来自已归档的输入技术预览。归一化显示和黄色ROI用于核对输入，不能作为医生诊断或夹层病灶真值；原预览中的`coverage pending`也不表示后来完成了临床覆盖认证。

#### 代表案例解析

| 案例与来源标签 | S1分数 / 本折阈值 | S1表现 | 对照解读 |
| --- | ---: | --- | --- |
| Dissec_02：来源阳性 | 0.666825 / 0.514373 | 正确识别（TP） | 三组均正确，单例不能代表所有阳性 |
| Dissec_06：来源阳性 | 0.409013 / 0.440865 | 漏判（FN） | 三组均漏判，预测ROI存在不保证识别正确 |
| Old_03：阴性候选 | 0.497256 / 0.514373 | 正确排除（TN） | S0+误判为阳性，S1纠正；不同阶段需使用各自验证阈值 |
| Old_02：阴性候选 | 0.952385 / 0.007967 | 误判（FP） | 三组均误判，仍需检查局部聚合、来源偏差和标签依据 |

分数不是校准后的患病概率；表内为四舍五入展示，原始预测依据完整精度分数和阈值。四例是结果后的描述性选择。Old_03的纠正使AortaSeg特异度从15/19变为16/19，但主比较AUROC差异区间仍跨0，不能据此宣称半监督整体有效。

[完整数据图、案例分数与注意力分析](engineering/主动脉夹层CTA半监督研究/项目实验报告.md#74-具体案例从输入阈值到分类结果) · [12条案例原始记录](engineering/主动脉夹层CTA半监督研究/results/case_examples.csv)

#### 整体结果与案例的关系

![CTA同来源评价与逐折结果](engineering/主动脉夹层CTA半监督研究/figures/03_as_results.png)

上图将同来源总体结果与逐折结果并列。S1的OOF AUROC为0.7263，S0+为0.7316，配对差异为−0.0053，95%区间[−0.0737, 0.0632]。案例展示了具体变化，总体分析则说明这些变化尚不足以形成明确改善证据。

本实践使用CTA进行夹层类别实验；平扫CT、其他AAS病变及临床多模态尚未在本实验中验证。详细报告保留来源标签、阴性候选、身份独立性和有限技术QC的解释范围。
