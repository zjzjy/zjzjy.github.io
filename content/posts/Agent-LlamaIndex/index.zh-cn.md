---
weight: 1
title: "LlamaIndex - Introduction"
subtitle: ""
date: 2025-06-19T14:38:53+08:00
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
本文根据[Hugging Face上的Agent课程](https://huggingface.co/learn/agents-course/unit2/llama-index/introduction)编写而成，包括        。
# 什么是LlamaIndex？
LlamaIndex 是一个完整的工具包，用于使用索引和工作流创建基于LLM的Agent。
## LlamaIndex的关键部分以及它们如何帮助代理？
**Components**：在 LlamaIndex 中使用的基本构建块。 These include things like prompts, models, and databases.组件通常用于将 LlamaIndex 与其他工具和库连接起来。  
**Tools**: 工具是提供特定功能（例如搜索、计算或访问外部服务）的组件。  
**Agents**：能够使用工具并做出决策的自主组件。它们协调工具的使用，以实现复杂的目标。  
**Workflows**：是将逻辑整合在一起的逐步流程。工作流或代理工作流是一种无需明确使用代理即可构建代理行为的方法。
## LlamaIndex的关键优势
- 清晰的工作流系统 ：工作流使用事件驱动和异步优先的语法，逐步分解代理的决策流程。这有助于您清晰地编写和组织逻辑。
- 使用 LlamaParse 进行高级文档解析 ：LlamaParse 是专为 LlamaIndex 制作的，因此集成是无缝的，尽管它是一项付费功能。
- 众多即用型组件 ：LlamaIndex 已经推出一段时间了，因此可以与许多其他框架兼容。这意味着它拥有许多经过测试且可靠的组件，例如 LLM、检索器、索引等等
- LlamaHub ：是数百个此类组件、代理和工具的注册表，您可以在 LlamaIndex 中使用它们。
# LlamaIndex的使用
## LlamaHub 简介
LlamaHub 是一个包含数百个集成、代理和工具的注册表，您可以在 LlamaIndex 中使用它们。
那么应该怎么使用呢？
LlamaIndex 的安装说明在 [LlamaHub](https://llamahub.ai/) 上提供了结构清晰的概述 。乍一看可能有点难以理解，但大多数安装命令通常都遵循一种易于记忆的格式 ：
```python
pip install llama-index-{component-type}-{framework-name}
```
让我们来尝试使用Hugging Face inference API integration安装 LLM 和嵌入组件的依赖项。
```python
pip install llama-index-llms-huggingface-api llama-index-embeddings-huggingface
```
使用刚刚下载好的组件的示例：
```python
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI
import os
from dotenv import load_dotenv

# Load the .env file
load_dotenv()

# Retrieve HF_TOKEN from the environment variables
hf_token = os.getenv("HF_TOKEN")

llm = HuggingFaceInferenceAPI(
  model_name="Qwen/Qwen2.5-Coder-32B-Instruct",
  temperature=0.7,
  max_tokens=100,
  token=hf_token,
)

response = llm.complete("Hello, how are you?")
print(response)
# I am good, how can I help you today?
```
## QueryEngine 组件
QueryEngine 组件可以用作代理的检索增强生成 (RAG) 工具。
![RAG](images/llamaindex_rag.png)
现在，想想 Alfred 是如何工作的：
- 你请阿尔弗雷德帮忙策划一场晚宴
- Alfred 需要检查你的日历、饮食偏好和过去成功的菜单
- QueryEngine 帮助 Alfred 找到这些信息并使用它来计划晚宴

现在，让我们更深入地了解组件，看看如何组合组件来创建 RAG 管道。
RAG包含五个关键步骤：
1. Loading：指的是将数据从其所在位置（无论是文本文件、PDF、其他网站、数据库还是 API）加载到您的工作流程中。LlamaHub 提供数百种集成方案供您选择。
2. Indexing：这意味着创建一个允许查询数据的数据结构。对于LLM来说，这几乎总是意味着创建向量嵌入。向量嵌入是数据含义的数值表示。索引还可以指许多其他元数据策略，以便于根据属性准确地找到上下文相关的数据。
3. Storing：一旦您的数据被索引，您将需要存储您的索引以及其他元数据，以避免重新索引它。
4. Querying：对于任何给定的索引策略，您可以通过多种方式利用 LLM 和 LlamaIndex 数据结构进行查询，包括子查询、多步骤查询和混合策略。
5. Evaluation：任何流程中的关键步骤是检查其相对于其他策略的有效性，或检查何时进行更改。评估可以客观衡量您对查询的响应的准确性、可靠性和速度。
### Loading and embedding documents
LlamaIndex 可以在您自己的数据上工作，但是， 在访问数据之前，我们需要加载它。 将数据加载到 LlamaIndex 主要有三种方法：
1. SimpleDirectoryReader ：用于从本地目录加载各种文件类型的内置加载器。
 ```python
from llama_index.core import SimpleDirectoryReader

reader = SimpleDirectoryReader(input_dir = "path/to/directory")
documents = reader.load_data()
 ```
这个多功能组件可以从文件夹中加载各种文件类型，并将它们转换为 LlamaIndex 可以使用的 Document 对象。  
加载文档后，我们需要将它们分解成更小的部分，称为 Node 对象。Node 只是原始文档中的一段文本，方便AI处理，同时仍然保留对原始 Document 对象的引用。
`IngestionPipeline` 通过两个关键转换帮助我们创建这些节点。
   - `SentenceSplitter` 按照自然句子边界将文档拆分为可管理的块。
   - `HuggingFaceEmbedding` 将每个块转换为数字嵌入 - 以 AI 可以有效处理的方式捕捉语义含义的矢量表示。
```python
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.ingestion import IngestionPipeline

# create the pipeline with transformations
pipeline = IngestionPipeline(
  transformations = [
    SentenceSplitter(chunk_overlap=0),
    HuggingFaceEmbedding(model_name = "BAAI/bge-small-en-v1.5"),
  ]
)

nodes = await pipeline.arun(documents=[Document.example()])
```
2. [LlamaParse](https://github.com/run-llama/llama_cloud_services/blob/main/parse.md) ：LlamaParse，LlamaIndex 用于 PDF 解析的官方工具，可作为托管 API 使用。
3. LlamaHub ：数百个数据加载库的注册表，用于从任何来源提取数据。

### Storing and indexing documents
创建 Node 对象后，我们需要对它们进行索引以使它们可搜索，但在执行此操作之前，我们需要一个地方来存储我们的数据。  
由于我们使用的是提取管道，因此可以直接将向量存储附加到管道来填充数据。在本例中，我们将使用 Chroma 来存储文档。  
可以在 [LlamaIndex 文档](https://docs.llamaindex.ai/en/stable/module_guides/storing/vector_stores/)中找到不同向量存储的概述。
```python
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore

db = chromadb.PersistentClient(path="./alfred_chroma_db") # 创建持久化客户端，将数据存储在本地文件系统（而非内存）。
chroma_collection = db.get_or_create_collection('alfred') #  获取或创建向量集合
vector_store = ChromaVectorStore(chroma_collection = chroma_collection) # 将 Chroma 集合包装为 LlamaIndex 兼容的向量存储接口。

pipeline = IngestionPipeline(
  transformations=[
    SentenceSplitter(chunk_size = 25, chunk_overlap = 0),
    HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5"),
  ],
  vector_store = vector_store,
)
```
通过将查询和节点嵌入 `VectorStoreIndex` 同一个向量空间中，我们可以找到相关的匹配项。  
让我们看看如何从向量存储和嵌入中创建这个索引。
```python
from llama_index.core import VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
index = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model)
```
所有信息都会自动保存在 `ChromaVectorStore` 对象和传递的目录路径中。
### Querying a VectorStoreIndex with prompts and LLMs
在查询索引之前，我们需要将其转换为查询接口。最常见的转换选项是：
- `as_retriever` ：用于基本文档检索，返回具有相似度分数的 NodeWithScore 对象列表
- `as_query_engine` ：对于单个问答交互，返回书面答复
```python
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
query_engine = index.as_query_engine(
  llm = llm,
  response_mode = "tree_summarize"#先从检索到的多个文档片段中生成局部摘要。再将这些摘要递归合并为最终答案，确保回答全面且连贯。
)
query_engine.query("What is the meaning of life?")
```
- `as_chat_engine` ：对于在多条消息中保持记忆的对话交互，使用聊天历史记录和索引上下文返回书面回复。
### Response Processing 
在底层，查询引擎不仅使用 LLM 来回答问题，还使用 ResponseSynthesizer 作为处理响应的策略。同样，这是完全可定制的，但主要有三种开箱即用的策略：
- refine ：通过按顺序遍历每个检索到的文本块来创建并优化答案。这会为每个节点/检索到的文本块单独调用一次 LLM。
- compact （默认）：类似于细化，但事先连接块，从而减少 LLM 调用。
- tree_summarize ：通过遍历每个检索到的文本块并创建答案的树形结构来创建详细的答案。
### Evaluation and observability
LlamaIndex 提供内置评估工具来评估响应质量。 这些评估人员利用 LLM 来分析不同维度的响应。 让我们看一下可用的三个主要评估器：
- FaithfulnessEvaluator ：通过检查答案是否得到上下文支持来评估答案的真实性。
- AnswerRelevancyEvaluator ：通过检查答案是否与问题相关来评估答案的相关性。
- CorrectnessEvaluator ：通过检查答案是否正确来评估答案的正确性。
```python
from llama_index.core.evaluation import FaithfulnessEvaluator

query_engine = # from the previous section
llm = # from the previous section

# query index
evaluator = FaithfulnessEvaluator(llm=llm)
response = query_engine.query(
    "What battles took place in New York City in the American Revolution?"
)
eval_result = evaluator.evaluate_response(response=response)
eval_result.passing
```
评估流程：
+ 分解回答：将回答拆分为多个独立的陈述（如 "Long Island 战役发生在纽约市" 和 "Fort Washington 战役发生在纽约市"）。
+ 检查依据：针对每个陈述，验证是否存在于检索到的上下文中。
+ 生成评估结果：
  + passing：布尔值，表示回答是否完全忠实于上下文。
  + score：分数（0-1），表示忠实程度。
  + feedback：详细反馈，指出不忠实的陈述及原因。
#### 安装 LlamaTrace
正如 LlamaHub 部分介绍的那样，我们可以使用以下命令从 Arize Phoenix 安装 LlamaTrace 回调：
```python 
pip install -U llama-index-callbacks-arize-phoenix

import llama_index
import os

PHOENIX_API_KEY = "<PHOENIX_API_KEY>"
os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
llama_index.core.set_global_handler(
    "arize_phoenix",
    endpoint="https://llamatrace.com/v1/traces"
)
```
# 创建实例
