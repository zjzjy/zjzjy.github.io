# Rec - HLLM Enhancing Sequential Recommendations via Hierarchical Large Language Models for Item and User Modeling

这篇论文试图解决的问题是如何在推荐系统中有效地利用大型语言模型（LLMs）来提升序列推荐的准确性和效率。具体来说，论文探讨了以下几个关键问题：
- 预训练权重的实际价值：研究预训练的大型语言模型（LLMs）中包含的世界知识和推理能力在推荐系统任务中的具体价值。

- 微调的必要性：探讨是否需要对推荐任务进行特定的微调（fine-tuning），以提高模型在推荐系统上的性能。

- 模型的可扩展性：验证大型语言模型在推荐系统中是否能够展现出与其他领域相似的良好扩展性，即模型性能是否随着模型参数数量的增加而提升。

论文通过提出一种新颖的层次化大型语言模型（HLLM）架构来解决上述问题，该架构通过两个层次的模型来分别提取项目特征和预测用户的未来兴趣。这种方法有效地利用了开源LLMs的预训练能力，并通过进一步的微调显著提高了性能。此外，HLLM在两个大规模数据集上的实验表明，它在训练和服务质量上都取得了良好的效果，并在在线A/B测试中验证了其在现实世界推荐场景中的实用影响。
## 基础背景
推荐的三个基础部分：
- Retriever（召回）：一般有多个召回模型，从不同多样的scource召回数据。比如一些rule-based召回，基于用户是否最近浏览、用户是否关注等等。大量的CG来源被整合作为召回层的输出。
- 排序：有粗排和精排，本质上是一样的，只是数据量大小不一样。输入是每一个item，输出是business metric。按b站举例，每个item输出点赞、转发、投币的概率。结合起来进行一个排序这是一个一对一的过程，一个item一个score。如果只有两层会有个什么问题呢？比如说你经常查看关于LLM的学术视频，那么系统可能会一直推送关于LLM和学术视频，但是用户可能会厌倦，我希望能推送一些多样化的视频。
- 重排：listwise，把所有item的list进行输入，进行整体的考虑重排。比如精排之后前面是个视频都是广告（因为广告权重比较大），那么在重排会减少广告视频的概率。

HLLM应用到召回和排序这两步。1.在召回这一步增加一个CG，推荐更加personalized内容，会更有潜意识的推荐，而不是根据过去历史直接推荐（这也是为什么选择使用LLM的原因，LLM可以通过一些word knowledge进行浅层学习。那word knowledge为什么不能通过其他召回层实现？因为召回层只见过自己的dataset，比如字节公司的召回只见过抖音、今日头条的数据，但是LLM用来pretrained data会有更广的数据，会有跳出平台的行为。）2.排序：针对于去看粗排/精排的模型的classification是不是更准。

之前也有尝试将LLM融入到推荐系统中，主要有三种approch：
1. RLMRec：将item的信息（title,description,...,user feedback等）用LLM来reasoning，special token去输出embedding。通过这些输出的embedding进行对比学习，
   ![RLMRec](/images/RLMRec.png)
2. User-LLM：把item的信息传入到encoder，将encoder的输出传入到decoder，并不会直接推荐下一个视频，而是通过问答的形式（用户更喜欢哪个类型？之类的问题）。用output跑sql做search。LLM用在Xformer Stack, 作为一个hidden states。
   ![User-LLM](/images/User-LLM.png)
3. LLARA：text token, behavior token（单独的feature extractor），通过一个实时训练的projector投到LLM的embedding上，concatenate一起做为LLM输入。
   ![LLARA](/images/LARA.png)

缺陷：成本难度比较大，难以落地。实时性达不到。

## 贡献
文章的三个主要贡献如下：
1. **提出了一个新颖的层次化大型语言模型（HLLM）框架**，用于序列推荐。该框架在大规模学术数据集上显著优于经典的基于ID的模型，并在现实世界的工业环境中得到了验证，展现出实际效益。此外，该方法还具有出色的训练和服务效率。
2. **有效地将预训练阶段编码的世界知识转移到推荐模型中**，涵盖了项目特征提取和用户兴趣建模。然而，针对推荐目标的任务特定微调是必不可少的。
3. **HLLM展现出优秀的可扩展性**，随着数据量和模型参数的增加，性能持续提升。这种可扩展性突显了该方法在应用于更大规模数据集和模型尺寸时的潜力。

## 模型
![HLLM-Fig1](/images/HLLM-fig1.png)
分为两个部分：1.item level的特征提取，通过item LLM。2. 学习用户和item交互的行为数据，也就是通过user item的history个iteraction history学习数据，通过user LLM。相比于双塔，这样设计考虑到了时序信息。  

Retrieval Part：
item LLM：输入是文本信息。会有个类似system prompt的task指引：compress the following sentence into embedding。最后加一个special token，special token的输出是我们想知道的embedding。

user LLM：每个输入都是由item LLM embedding来的。

loss相当于一个对比学习，正样本是预测的$E'_{n+1}$，负样本是pool中ID不相同的item的embedding。
这里注意到没有使用user profile，也 make sense。消除了刻板印象，用交互行为作为用户的profile。
Ranking Part：
![HLLM-Fig2](/images/HLLM-Fig2.png)
item LLM不再训练了，user LMM重新训练。  
Early Fusion：target是从pool（candidate）里拿的每个项目，查看在这个用户的行为下，target是a这个item的转化率、点击率等为多少。
Late Fusion：在最后做了一个user special token得到一个embedding，与target的embedding结合，通过一个MLP输出logits。
**区别**：Early Fusin比较慢，Late Fusion生成的user embedding只生成一次就可以。一般来说，early fusion效果更好。不太好做实时。
**Loss**：multitask loss + 权重*第一部分autoregression loss
## 实验细节
回答以下问题：
1. LLM在文本上pretrain的知识是否能够迁移到recommandation上？
2. 有没有scaling law？
3. HLLM比起其他方法由哪些优势？
4. 如何实现train和inference？

### Retriever
Dataset: Pixel8M(max sequence length 10), Amazon Book Reviews(max sequence length 50).
trick1: max sequene length太小了，做的和其他模型的对比实验也是控制了max sequence length 10 or 50。

**EX1**：在文本上pretrain再迁移到rec会不会更好？
A： 从表2的结果看，是的。

**EX2**： 见过文本的数量会不会影响？
A：从表3来看，见过token更多效果更好。注意到，1T+chat的效果比较差。1T+chat 表示在预训练阶段使用了1万亿个token进行训练，并且在微调阶段对对话数据进行了监督式微调。这里的 +chat 表示在微调阶段特别针对对话数据进行了优化。某种意义上，可能说明了SFT作为对齐的一种方法，不适用于迁移任务。表四表示item llm和user llm上都是可学习的参数更新微调，效果最好（item llm更核心，适合任务的embedding space比较重要，这也说明了如果直接拿大模型直接用的话效果不会太好，需要在自己的任务上进行微调）。

**EX3**：user-llm和item-llm参数大小的影响？
A：表5，随着更好的item llm，会有更好的表现。表6同样也有相似的表现，但是差距很小。

**EX4**：数据量的影响？
A：图三，数据更大，效果更好。跟HSTU对比，但还是sequence 10的。虽然说HSTU 1B模型，但是大部分参数（600M左右）在embedding table上，实际上可以算400M。

**EX5**：比起其他模型？
A：表7，大多数模型都是固定max seq，作者重新训练的。HSTU-large(2024)的数据并非重新训练，而是直接拿的HSTU论文中的数据。Amazon Books来说，max seq=50能carry所有数据。

这里有一点值得注意，retriever并没有一个public数据量比较大的数据集。

**EX6**：在一部分数据集上联合训练user-llm和item-llm和在全部数据集上freeze item-llm，只训练user-llm，这样子会好吗？（目的是在seq很长的情况下，由于memory limits，不可能handle所有的数据）
A：表8，确实也可以表现得不错。
### Ranking
没有public dataset，大家都在通过online A/B test进行测试 ，在关键指标上绝对值提升0.7%。
![Online](/images/HLLM-online.png)
训练：endtoend，联合训练。sequence length截到150。为了在更长得sequence上有更好得表现，把item llm freeze，只在user llm上训练，把150拉到1000（直接在item-llm提取得特征上训练）。

Inference：为什么能做的块？没有在online的时候inference，而是把item llm, user llm做成embedding的提取器。当一个item被创造出来，call item llm生成这个item的embedding，save。每一天，update user的时序信息的embedding存到datavase。实际上得到两个表，在做ranking的时候，把这两个表拿进来做MLP。所以只有MLP，非常快。

### 补充材料
表10：怎么更好地提取item的embedding？ average pooling    和   item special token（自回归）。

表11：在public数据集上增大sequence会更好吗？会

表12：尝试不同的embedding？LLM embedding+timestep 表现更好。  item ID反而表现更差。

表13，14：sequence length越大，表现越好；模型越大，表现越好。item影响更大。

内容来自[b站up主酸果酿](https://www.bilibili.com/video/BV1uXoBY3E8B?spm_id_from=333.788.videopod.sections&vd_source=dcd6c275fe4ed979bb96cd340654e13c)。

