---
weight: 1
title: "Rec - Sparse Meets Dense: Unified Generative Recommendations with Cascaded Sparse-Dense Representations"
subtitle: ""
date: 2025-07-03T14:22:30+08:00
draft: true
author: "June"
authorLink: "https://github.com/zjzjy"
description: ""
images: []
resources:
- name: "featured-image"
  src: "featured-image.jpg"

tags: []
categories: []
lightgallery: true

hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: ""

license: '<a rel="license external nofollow noopener noreffer" href="https://creativecommons.org/licenses/by-nc/4.0/" target="_blank">CC BY-NC 4.0</a>'
toc:
  auto: false
---
创新点：
- **COBRA框架**：创新性地整合稀疏语义标识符和密集向量，通过级联表示捕捉项目语义和细粒度特征。
- **从粗到精策略**：推理时先生成稀疏标识符定位项目类别，再生成密集向量捕捉细节，提升推荐准确性和个性化。
- **BeamFusion机制**：结合beam search和最近邻检索分数，平衡推荐的精确度和多样性，增强系统灵活性。
- **端到端训练**：动态优化密集表示，捕捉用户 - 项目交互的语义和协同信号，适应推荐需求。
## 实施细节
![COBRA概览](/assets/images/COBRA-outline.png)
COBRA的输入是一系列级联表示，由稀疏ID和与用户交互历史中的项相对应的密集向量组成。在训练过程中，密集表示是通过对比学习目标和端到端的方式学习的。通过首先生成稀疏ID，然后生成稠密表示，COBRA降低了稠密表示的学习难度，并促进了两种表示之间的相互学习。在推理过程中，COBRA采用了一个由粗到细的生成过程，从稀疏ID开始，稀疏ID提供了一个捕获项目分类本质的高级分类草图。然后，生成的ID被附加到输入序列中，并反馈到模型中，以预测捕获细粒度细节的密集向量，从而实现更精确和个性化的推荐。为了确保灵活的推理，我们引入了BeamFusion，这是一种将波束搜索与最近邻检索分数相结合的采样技术，确保了检索到的项目的可控多样性。与TIGER不同，它只依赖于稀疏ID，COBRA利用稀疏和密集表示的优势。
- `‘密集表示是通过对比学习目标和端到端的方式学习的’怎么理解？`：
- `怎么就降低了密集表示生成的难度？`：
- `怎么就促进了两种表示之间的相互学习？`：
- `既然都要生成两种表示，怎么会降低计算复杂度？`：
- 
### Sparse-Dense Representation
![COBRA细节](/assets/images/COBRA-details.png)
**Sparse Representation**：对于每个项目，我们提取其属性以生成文本描述，该文本描述被嵌入到密集向量空间中并被量化以生成稀疏ID。这些ID捕获了项目的分类本质，形成了后续处理的基础。为了简洁起见，随后的方法描述将假设稀疏ID由单个级别组成。
- `提取其属性以生成文本描述，该文本描述被嵌入到密集向量空间中并被量化以生成稀疏ID。`：例如，对于一个商品项目，其属性可能包括品牌、类型、价格范围、主要功能等。这些属性被提取出来后，可以形成一段描述该商品的文本，如“这是一款苹果品牌的智能手机，价格在5000-6000元之间，具有高清摄像头和快速充电功能”。使用预定义的模板将提取的属性填充到模板中，生成文本描述。再将这段描述映射到高维空间。

**Dense Representation**：为了捕获细微的属性信息，我们开发了一个端到端的可训练密集编码器，对项目文本内容进行编码。每个项目的属性都被展平为一个文本语句，并以[CLS]标记作为前缀，然后被送入基于Transformer的文本编码器Encoder。密集表示v是从对应于[CLS]标记的输出中提取的，捕获了项的文本内容的细粒度细节。如图所示，结合了位置嵌入和类型嵌入来对序列中标记的位置和上下文进行建模。这些嵌入以相加的方式添加到令牌嵌入中，增强了模型区分不同标记及其在序列中的位置的能力。

**Cascaded Representation**：将稀疏ID和密集向量集成在统一的生成模型中。稀疏ID通过离散约束提供了稳定的分类基础，而密集向量保持了连续的特征分辨率，确保模型既能捕获高级语义，又能捕获细粒度细节。
-`'稀疏ID通过离散约束提供了稳定的分类基础，而密集向量保持了连续的特征分辨率，确保模型既能捕获高级语义，又能捕获细粒度细节'，怎么做到的？`
### Sequential Modeling
**Probabilistic Decomposition**目标项的概率分布建模被分解为两个阶段，利用稀疏表示和密集表示的互补优势。具体来说，COBRA不是直接根据历史交互序列$S_{1:t}$预测下一个项目$s_{t+1}$，而是分别预测稀疏ID $ID_{t+1}$和密集向量$v_{T+1}$：  
$P(ID_{t+1},v_{t+1}|S_{1:t})=P(ID_{t+1}|S_{1:t})P(v_{t+1}|ID_{t+1},S_{1:t})$  
其中，$P(ID_{t+1}|S_{1:t})$表示基于历史交互序列$S_{1:t}$预测$ID_{t+1}$，即捕获下一项的分类本质。$P(v_{t+1}|ID_{t+1},S_{1:t})$则捕获下一项的细颗粒细节。
**Sequential Modeling with a Unified Generative Model**
- Embedding Sparse IDs：稀疏ID，通过emdedding层转化为dense vector：$e_{t} = Embed(ID_{t})$，连接$e_{t},v_{t}$形成模型再每个时间步的输入:  
    $h_t=[e_t:v_t].$
- Tansformer Modeling：Transformer Decoder模型包括多个层，每个层都具有自注意机制和前馈网络。如图上半部分所示，解码器的输入序列由级联表示组成。为了增强序列和上下文信息的建模，这些表示被增加了项目位置和类型嵌入。为了简洁起见，下面几节中的数学公式集中在级联序列表示上，省略了位置和类型嵌入的显式符号。解码器处理该丰富的输入以生成用于预测后续稀疏ID和密集向量的上下文化表示。
- Sparse ID Prediction：给定历史交互序列$S_{1:t}$预测稀疏$ID_{t+1}$，transformer的输入序列是：  
  $S_{1:t}=[h_1,h_2,...,h_t]=[e_1,v_1,e_2,v_2,...e_t,v_t].$  
  Transformer decoder处理$S
_{1:t}$产生一系列向量$y_t=TransformerDecoder(S_{1:t})$，稀疏ID的预测推导为：  
$z_{t+1}=SparseHead(y_t)$   
其中，$z_{t+1}表示$ID_{t+1}的概率。$$
- Dense Vector Prediction：为了预测$v_{t+1}$，Transformer 输入序列为：  
- $\bar{S}_{1:t}=[S_{1:t},e_{t+1}]=[e_1,v_1,e_2,v_2,...e_t,v_t,e_{t+1}]$  
- Transformer decoder处理$\bar{S}_{1:t}$产生对于$v_{t+1}$的预测：  
  $\hat{v}_{t+1}=TransformerDecode(\bar{S}_{1:t})$
### End-to-End Training
端到端训练过程旨在联合优化稀疏和密集表示预测。训练过程由组合稀疏ID预测和密集向量预测的损失的复合损失函数控制。稀疏ID预测损失：$L_{sparse}=-\Sigma_{t=1}^{T-1}log(\frac{exp(z_{t+1}^{ID_{t+1}})}{\Sigma_{j=1}^{C}exp(z^{j}_{t+1})})$。密集向量预测损失$L_{dense}$专注于细化密集向量，使它们能够区分相似和不相似的项目,$L_{dense} = -\Sigma_{t=1}^{T-1}log\frac{exp(cos(\hat{v}_{t+1}\dot v_{t+1}))}{\Sigma_{item_{j\in Batch}exp(cos(\hat{v}_{t+1},v_{item_{j}}))]}}$。密集矢量由端到端可训练编码器Encoder生成，该编码器在训练过程中进行了优化。这确保了密集向量被动态地细化并适应推荐任务的特定要求。
## Coarse-to-Fine Generation

### 想要解决什么问题？

## 如何解决的？

## 实验结果解读
