---
weight: 1
title: "Rec - Large Scale Product Graph Construction for Recommendation in E-commerce"
subtitle: ""
date: 2025-07-28T10:08:40+08:00
draft: false
author: "June"
authorLink: "https://github.com/zjzjy"
description: ""
images: []
resources:
- name: "featured-image"
  src: "featured-image.jpg"

tags: ["Rec"]
categories: ["Rec"]
lightgallery: true

hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: ""

license: '<a rel="license external nofollow noopener noreffer" href="https://creativecommons.org/licenses/by-nc/4.0/" target="_blank">CC BY-NC 4.0</a>'
toc:
  auto: false
---

## 这篇论文想要解决什么问题？
解决**大规模电商推荐系统中产品关系图构建**的关键挑战。
论文关注的核心问题包括：

- **产品间关系的捕捉**：理解并捕捉产品之间的关系是现代电子商务推荐系统的基础。这些关系可以被视为产品重构索引，能够根据给定的种子产品返回排名列表。

- **产品替代和互补关系**：产品间存在两种非常重要的关系：替代关系和互补关系。替代产品是可以互相替换的产品，而互补产品是可能一起购买的产品。不同环境对推荐的相关性有不同的要求，因此需要不同的关系图来加速推荐。

- **挑战**：大规模推荐系统面临的挑战包括准确性、鲁棒性、数据稀疏性、关系方向性和可扩展性。

为什么需要解决这些问题？
1. 电商业务驱动：
  - 规模：淘宝/Amazon 等平台每日需服务 数十亿用户、千亿级商品，传统推荐算法无法在 O(1)时间复杂度 内完成实时预测（如毫秒级响应）。
  - 场景复杂性：用户行为分 “购买前”（需要替代品比较） 和 “购买后”（需要互补品搭配），单一关系图无法满足动态需求。
  - 商业指标：推荐延迟 >200ms 会显著降低转化率（CTR/CVR），需通过 预构建产品图索引 加速召回。
2. 数据挑战：
  - 噪声：用户点击中存在大量误触、刷单等噪声，传统方法无法区分。
  - 稀疏性：购买行为稀疏（点击/购买比例≈100:1），互补关系更难捕捉。
  - 方向性：互补关系是 非对称 的（手机→手机壳合理，反向不合理）。

之前的算法以及优缺点：
| **类别**        | **代表算法**                                                           | **优点**                 | **缺点**（论文针对的痛点）                                                                                     |
| ------------- | ------------------------------------------------------------------ | ---------------------- | --------------------------------------------------------------------------------------------------- |
| **协同过滤 (CF)** | - Item-based CF  \[Sarwar’01]  <br> - User-based CF \[Breese’98]   | - 简单、易并行 <br> - 无需内容特征 | - **仅局部相似度**（如余弦/Jaccard），忽略用户-商品图的高阶结构 <br> - **噪声敏感**：误点击直接干扰相似度计算 <br> - **稀疏性**：共购数据稀疏，互补关系难以捕捉 |
| **内容推荐 (CB)** | - TF-IDF 向量 \[Pazzani’97] <br> - 词袋模型 \[Mooney’00]                 | - 解决冷启动 <br> - 可解释性强   | - **文本特征噪声大**：淘宝商品标题/描述由商家填写，质量参差不齐 <br> - **无法捕捉隐含需求**：如“红色T恤”与“红色运动鞋”的互补关系                        |
| **混合方法**      | - Amazon Item-to-Item \[Linden’03] <br> - Hybrid CF+CB \[Burke’02] | - 结合多源信息               | - **复杂度高**：需融合文本、行为、上下文，难以实时计算 <br> - **扩展性差**：O(N×M) 复杂度无法应对十亿级数据                                  |
| **图方法**       | - 随机游走 \[Koren’10] <br> - 矩阵分解 \[Rendle’10]                        | - 捕捉高阶关系 <br> - 支持时序   | - **三阶以上计算爆炸**：无法实时 <br> - **稀疏性未解决**：需大量采样或正则化                                                     |

{{< admonition type=note title=ItemCF open=false >}}
[王树森老师的b站视频](https://www.bilibili.com/video/BV1mA4y1Q7RN/?spm_id_from=333.337.search-card.all.click&vd_source=dcd6c275fe4ed979bb96cd340654e13c)
例子： “我喜欢看《笑傲江湖》” + “《笑傲江湖》与《鹿鼎记》相似”+“我没看过《鹿鼎记》” ——> 给我推荐《鹿鼎记》。逻辑很简单，问题在推荐系统如何知道《笑傲江湖》与《鹿鼎记》相似？

量化用户对交互过的物品的兴趣，已知物品之间的相似度，预估用户对候选物品的兴趣：$$\Sigma like(user,item_{j}) sim(item_{j},item)$$

如何计算物品相似度？- 受众重合度越高，物品越相似 $sim(i_1,i_2) = \frac{|V|}{\sqrt{|W_1| |W_2|}}$；考虑用户喜爱程度：$$sim(i_1,i_2)=\frac{\Sigma like(v,i_1) like(v,i_2)}{\sqrt{\Sigma like^{2}(u_1,i_1)}\sqrt{\Sigma like^{2}(u_2,i_2)}}$$余弦相似度，把每个物品表示为一个稀疏向量，每个元素对应一位用户。

召回流程：
1. 事先做离线计算
   1. 建立“用户->物品”的索引：记录每个用户最近点击、交互过的物品ID，这样给定任意用户ID，可以找到他近期感兴趣的物品列表。
   2. 建立“物品->物品”的索引：计算物品之间凉凉相似度。对于每个物品，索引它最相似的k个物品。
2. 线上做召回
   1. 给定用户ID，通过“用户->物品”索引，找到用户近期感兴趣的物品列表(last-n)
   2. 对于last-n列表中每个物品，通过“物品->物品”的索引，找到top-k的相似物品。
   3. 对于取回的相似物品（最多有nk个），用公式预估用户对物品的兴趣分数
   4. 返回分数最高的100个物品，作为推荐结果。

Q：索引的意义在于避免枚举所有的物品。
Q：离线计算大（更新索引），在线计算小。
{{< /admonition >}}
{{< admonition type=note title=TF-IDF open=false >}}
term frequency–inverse document frequency
TF-IDF的主要思想是：如果某个单词在一篇文章中出现的频率TF高，并且在其他文章中很少出现，则认为此词或者短语具有很好的类别区分能力，适合用来分类。
{{< /admonition >}}
{{< admonition type=note title=随机游走 open=false >}}
n2v这些图的算法，本质上就是特征提取器，对graph进行采样（Sampling），对采出来的序构建模型（Embedding），最终把节点转化为特征向量，即embedding。
{{< /admonition >}}
## 论文如何解决这些问题？
### Swing
传统 CF 像 “数共同好友” 来判断两人是否相似，而 Swing 像 “检查共同好友的朋友圈是否也重叠。

“把『用户-商品-用户』这条二阶结构当成一个『稳固小三角形』，每出现一次就给商品对加分；用户之间重叠商品越多（越泛化），权重越低，最终累加得到抗噪的替代相似度。”

Swing 算法 3 步流程（对应论文 2.1）  
1. 构图  
   • 输入：用户-点击商品列表  
   • 输出：无向二部图 G=(U∪I,E)。  

2. 计算相似度（公式 1）  
   • 对任意商品对 (i,j)，取共同点击用户集合 Uᵢ∩Uⱼ。  
   • 对其中每对用户 (u,v)，算  
     s(i,j)=Σ Σ 1/(α+|Iᵤ∩Iᵥ|)。  
   • α=1 默认，|Iᵤ∩Iᵥ| 越大→权重越小。  

3. 用户活跃度惩罚（公式 2）  
   • 引入 wᵤ=1/√|Iᵤ|，最终  
     s(i,j)=Σ Σ wᵤ·wᵥ·1/(α+|Iᵤ∩Iᵥ|)。  
   • 越活跃的用户贡献越小。  

复杂度：O(T·N²·M)，论文用 MapReduce/Spark 并行化。

并行实现细节（对应论文 2.2）  
1. 数据格式  
   • 每行：user → [i₁,i₂,…,iₙ]（该用户点过的全部商品）。  

2. Map 阶段  
   • 对每个 user 的每条边 (u,i)，广播到所有邻居，输出  
     key = (i, j) 或 (j, i)，value = u。  

3. Reduce 阶段  
   • 按 key=(i,j) 聚合得到共同用户列表 Uᵢ∩Uⱼ。  
   • 在本地计算 Swing 值，输出 (i,j,s(i,j))。  

4. 工程优化  
   • Combiner 提前聚合，减少 shuffle 量；  
   • 爆款商品采样高频用户，防止 reducer 内存爆炸。  

Q：如果两人越同质，那么在相似度上的贡献权重应该越大？
A：弱化泛化用户
Q：确实有泛化用户的存在，但这样会不会错过真正很像的用户？
A：用户权重惩罚项

### **Swing 算法：严谨定义与公式**

#### **1. 图模型**
给定用户-商品二部图 \( \mathcal{G} = (\mathcal{U} \cup \mathcal{I}, \mathcal{E}) \)，其中边 \( (u, i) \in \mathcal{E} \) 表示用户 \( u \in \mathcal{U} \) 对商品 \( i \in \mathcal{I} \) 的点击行为。

#### **2. 相似度定义**
对于任意商品对 \( (i, j) \in \mathcal{I} \times \mathcal{I} \)，定义共同点击用户集合：
\[
\mathcal{S}_{ij} = \mathcal{U}_i \cap \mathcal{U}_j
\]
其中 \( \mathcal{U}_i = \{ u \in \mathcal{U} \mid (u, i) \in \mathcal{E} \} \) 为点击商品 \( i \) 的用户集合。

**Swing 相似度**定义为：
\[
\text{Swing}(i, j) = \sum_{u \in \mathcal{S}_{ij}} \sum_{\substack{v \in \mathcal{S}_{ij} \\ v \neq u}} \frac{w_u w_v}{\alpha + |\mathcal{I}_u \cap \mathcal{I}_v|}
\]
其中：
- \( \mathcal{I}_u = \{ i \in \mathcal{I} \mid (u, i) \in \mathcal{E} \} \) 为用户 \( u \) 点击的商品集合；
- \( w_u = \frac{1}{\sqrt{|\mathcal{I}_u|}} \) 为用户活跃度惩罚权重（Adamic/Adar 形式）；
- \( \alpha > 0 \) 为平滑超参数（论文中默认 \( \alpha = 1 \)）。

#### **3. 复杂度分析**
设：
- \( |\mathcal{I}| = T \)（商品总数）；
- 平均商品度为 \( N \)（每商品平均被 \( N \) 个用户点击）；
- 平均用户度为 \( M \)（每用户平均点击 \( M \) 个商品）。

则 Swing 计算复杂度为：
\[
\Theta(T N^2 M)
\]
通过 **MapReduce/Spark** 两级聚合实现线性可扩展：
- **Map 阶段**：广播用户点击向量；
- **Reduce 阶段**：局部计算商品对相似度。

#### **4. 关键性质**
- **噪声鲁棒**：需至少两个用户共同点击商品对，误点击影响指数下降；
- **长尾友好**：仅依赖局部二阶共现，无需全局统计；
- **可并行**：无共享状态，适合分布式计算。
### Surprise


#### **1. 问题定义**  
给定用户-商品购买记录，学习互补关系  
\[
\mathcal{G}_\text{purchase} = (\mathcal{U} \cup \mathcal{I}, \mathcal{E}_\text{purchase}),
\]  
目标：为每件已购商品 \(i\) 输出互补候选商品列表，解决**数据稀疏**与**时间敏感性**。

---

#### **2. 三级漏斗框架**

| 层级 | 输入 | 公式 | 作用 |
|---|---|---|---|
| **类别级** | 商品类别树 & 共购 | \(\theta(c_i, c_j) = \frac{N(c_j \succ c_i)}{N(c_j)}\) | 先筛大类，降候选量 |
| **商品级** | 时序购买记录 | \(s_1(i,j) = \frac{\sum_{u \in \mathcal{U}_{i \succ j}} \frac{1}{1+|t_{u,j}-t_{u,i}|}}{|\mathcal{U}_i|\cdot|\mathcal{U}_j|}\) | 细粒度互补，带时间衰减 |
| **聚类级** | Swing 相似度图 | \(s_2(i,j) = s_1\bigl(L(i),L(j)\bigr)\) | 用簇-簇共购补稀疏 |

---

#### **3. 核心公式**

1. **类别相关度**  
\[
\theta(c_i, c_j) = \frac{\left|\{u \mid \exists t_{u,j} > t_{u,i},\, i \in c_i,\, j \in c_j\}\right|}{\left|\{u \mid j \in c_j\}\right|}
\]

2. **商品级互补分数**（含时间衰减）  
\[
s_1(i,j) = \frac{\displaystyle\sum_{u \in \mathcal{U}_{i \succ j}} \frac{1}{1+|t_{u,j}-t_{u,i}|}}{|\mathcal{U}_i|\cdot|\mathcal{U}_j|}, \qquad \mathcal{U}_{i \succ j} \triangleq \{u \mid t_{u,j} > t_{u,i}\}
\]

3. **聚类级互补分数**  
\[
s_2(i,j) = \frac{\displaystyle\sum_{u \in \mathcal{U}_{L(i) \succ L(j)}} \frac{1}{1+|t_{u,L(j)}-t_{u,L(i)}|}}{|\mathcal{U}_{L(i)}|\cdot|\mathcal{U}_{L(j)}|}
\]

4. **最终 Surprise 得分**  
\[
s(i,j) = \omega \cdot s_1(i,j) + (1-\omega)\cdot s_2(i,j), \qquad \omega=0.8\ (\text{实验设定})
\]

---

#### **4. 聚类实现（Algorithm 2）**  
- **图**：以 Swing 相似度为权重的有向图 \(G=(\mathcal{I},E)\)。  
- **算法**：Label Propagation（异步更新，阻尼系数 \(\beta=0.25\)，最大迭代 10 轮）。  
- **复杂度**：\(O(|E|)\)，十亿节点 15 min 收敛。

---

#### **5. 复杂度与效果**  
- **离线**：共购-CF 1.3 h → Surprise 2.5 h（可接受）。  
- **在线 A/B**：  
  - CTR +35 %、CVR +183 %（vs. 共购-CF）。  
  - 长尾商品覆盖提升 2.1×。

---

#### **6. 性质**  
- **稀疏鲁棒**：簇级统计 + 类别过滤 + 时间衰减三重降稀疏。  
- **方向保持**：严格 \(t_{u,j} > t_{u,i}\) 保证互补方向性。  
- **可扩展**：MapReduce/Spark 并行，支持千亿边图。