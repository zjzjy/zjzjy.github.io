---
weight: 1
title: "Rec - Recommender Systems with Generative Retrieval"
subtitle: ""
date: 2025-06-26T14:18:22+08:00
draft: false
author: "June"
authorLink: "https://github.com/zjzjy"
description: ""
images: []
resources:
- name: "featured-image"
  src: "featured-image.jpg"

tags: ["GenRec"]
categories: ["GenRec"]
lightgallery: true

hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: ""

license: '<a rel="license external nofollow noopener noreffer" href="https://creativecommons.org/licenses/by-nc/4.0/" target="_blank">CC BY-NC 4.0</a>'
toc:
  auto: false
---
## Related Work
**Sequential Recommenders**：序列推荐系统旨在根据用户过去的交互行为序列来预测用户接下来可能与之交互的项目。早期方法常依赖于马尔可夫链技术来基于历史交互建模用户行为。近年来，基于Transformer的模型被广泛应用于序列推荐系统中，这些模型能够捕捉用户交互序列中的长距离依赖关系。在使用时，首先需要收集用户与项目的交互数据，并按照时间顺序构建用户的行为序列。然后，选择合适的序列模型（如GRU、LSTM、Transformer等）来学习用户的行为模式。在训练过程中，模型会根据用户的历史行为序列来预测用户接下来可能感兴趣的项目。最后，在实际应用中，根据模型的预测结果为用户生成推荐列表。  
- [GRU4REC](https://arxiv.org/abs/1511.06939)：本文提出的TIGER框架在序列推荐任务中，借鉴了GRU4REC的思想，但采用了更先进的Transformer架构来建模用户的行为序列，从而提高了推荐性能。
- [SASRec](https://arxiv.org/abs/1808.09781)：TIGER框架在建模用户行为序列时，借鉴了SASRec中自注意力机制的思想，但进一步通过生成式检索方法直接预测项目的语义ID，从而实现了更高效的检索和推荐。
- [BERT4Rec](https://arxiv.org/pdf/1904.06690v2)：TIGER框架在建模用户行为序列时，借鉴了BERT4Rec中Transformer架构的思想，但通过生成式检索方法直接预测项目的语义ID，从而实现了更高效的检索和推荐。
  
**Semantic IDs**：语义ID是一种用于表示项目的序列，它由一系列离散的标记组成，能够捕捉项目的语义信息。生成语义ID的过程通常包括两个步骤：首先，使用预训练的文本编码器（如BERT、Sentence-T5等）将项目的文本描述编码为嵌入向量；然后，通过自编码器框架（如RQ-VAE）对嵌入向量进行量化，生成语义ID。在推荐系统中使用语义ID时，首先需要为数据集中的每个项目生成语义ID。然后，在训练推荐模型时，将项目的语义ID作为输入，而不是传统的项目ID。这样，模型能够学习到项目之间的语义相似性，从而提高推荐的准确性和多样性。此外，语义ID的层次化特性还可以用于解决推荐系统中的冷启动问题。
- [VQ-Rec](https://arxiv.org/abs/2210.12316)：TIGER框架在生成语义ID时，借鉴了VQ-Rec中矢量量化技术的思想，但采用了更先进的RQ-VAE方法来生成层次化的语义ID，从而提高了推荐的准确性和多样性。
- [RQ-VAE](https://arxiv.org/abs/2107.03312)：TIGER框架在生成语义ID时，采用了RQ-VAE方法来量化项目的嵌入向量，从而生成层次化的语义ID。这种方法不仅提高了推荐的准确性和多样性，还减少了存储需求。

**Generative Retrieval**：生成式检索是一种信息检索方法，它利用生成模型直接预测目标项目的索引。与传统的基于嵌入的检索方法不同，生成式检索不需要为每个项目生成和存储单独的嵌入向量。在生成式检索中，项目通过语义ID来表示，模型通过自回归的方式预测用户接下来可能交互的项目的语义ID。在推荐系统中实现生成式检索时，首先需要训练一个生成模型（如基于Transformer的序列到序列模型），该模型能够根据用户的历史行为序列生成目标项目的语义ID。在检索阶段，模型会根据用户的交互历史自回归地解码目标项目的语义ID，然后通过查找表将语义ID映射回实际的项目。此外，通过调整生成过程中的温度参数，可以控制推荐结果的多样性。
- [DSI](https://arxiv.org/abs/2202.06991)：DSI是第一个将端到端Transformer用于检索应用的系统。它通过为每个文档分配结构化的语义DocID，并在给定查询时自回归地返回DocID，实现了高效的检索。TIGER框架在生成式检索方面，借鉴了DSI中Transformer用于检索的思想，但进一步通过生成项目的语义ID来实现更高效的检索和推荐。
- <span style="background: #d4fcbc; color: black;">概念理解：什么叫直接预测目标项目的索引？什么叫“不需要为每个项目生成和存储单独的嵌入向量？</span>
  在传统的基于嵌入的检索方法中，推荐系统通常会将查询（用户的意图）和候选项目都映射到同一个高维空间中。然后，通过计算查询嵌入和候选项目嵌入之间的相似度（如余弦相似度或内积），来找到最相关的候选项目。这种方法需要为每个项目生成一个嵌入向量，并将这些嵌入向量存储在一个索引结构中，以便快速检索。
  而在生成式检索中，模型的目标是直接生成目标项目的索引（或标识符）。这里的“索引”可以理解为项目的唯一标识符，例如项目的ID。模型通过学习用户行为序列的模式，直接预测用户接下来可能交互的项目的ID，而不是先生成嵌入向量再进行相似度计算。这种方法避免了为每个项目生成和存储嵌入向量的需要，从而减少了存储需求。
- <span style="background: #d4fcbc; color: black;">概念理解：什么叫以自回归方式预测？</span>
  自回归（Autoregressive）是一种生成模型的常见方法，它通过逐步生成序列中的下一个元素来构建输出。在生成式检索中，模型会根据用户的历史行为序列，逐步生成目标项目的索引。

## 框架介绍
![TIGER Overview](images/genrec-google-overview.png)
1. **生成语义ID**：这一步涉及将项目的内容特征（例如标题、描述或图片）编码成嵌入向量，然后通过量化方案将这些嵌入向量转换成codewords元组，具体来说：
  - 使用预训练的内容编码器（如 Sentence-T5 或 BERT）来生成项目的语义嵌入。
  - 应用量化方法（例如 RQ-VAE）来生成语义ID，这通常涉及将连续的嵌入向量映射到离散的代码字空间。
2. **训练生成式推荐系统**：这一步涉及使用基于 Transformer 的序列到序列模型，并在语义ID上进行训练，以预测用户接下来可能交互的项目。
## Semantic ID 怎么生成？
![RQ-VAE Overview](images/RQ-VAE.png)
**RQ-VAE**：通过更新codebook和DNN encoder-decoder参数共同训练自动编码器。
RQ-VAE首先通过编码器$\mathcal{E}$编码输入x，以学习潜在表示z：= $\mathcal{E}$（x）。在d = 0时，初始残差简单地定义为r0：= z。在每个level的d中，我们都有一个CodeBook $\mathcal{C}_d$：= $\{e_k\}_{k=1}^{K}$ = 1，其中k是codebook的大小。然后，通过将r0映射到该level的codebook中closest的嵌入来量化r0。 d = 0时，$e_{c_d}$的索引，即$c_0 = arg min_{i}∥r0-e_{k}∥$。对于下一个d = 1，残差定义为r1：= r0  -  $e_{c_0}$。然后，类似于level 0，通过找到最接近r1的嵌入方式来计算第一级的代码。该过程是递归的M次重复，以获取代表语义ID的M codeWord元组。这种递归方法近似于从粗到细粒度的输入。请注意，我们选择为每个M级别使用大小K的单独代码簿，而不是使用单个MK大小的代码簿。之所以这样做，是因为residue的范数倾向于随着水平的增加而降低，因此允许不同水平的不同粒度。  
我们有语义ID$(c_0,...,c_{m-1})$，z的量化表示为$\hat{z}:=\Sigma^{m-1}_{d=0}e_{c_i}$，再把$\hat{z}$传递给decoder。RQ-VAE的loss定义为$\mathcal{L}(x):=\mathcal{L}_{recon}+\mathcal{L}_{rqvae},\text{     } \mathcal{L}_{recon}:=∥x-\hat{x}∥^2,\mathcal{L}_{rqvae}:=\Sigma^{m-1}_{d=0}∥sg[r_i]-e_{c_i}∥^2+\beta∥r_i-sg[e_{c_i}]∥^2$。sg是stop-gradient，以做到分部分不打扰地参数更新。基于K-Means聚集的初始化为codebook。  
实验细节
## 做了哪些实验？

## 解决什么问题？怎么解决的？

## 有什么未来改进方向？
