---
weight: 1
title: "Agentic RAG - Usecase"
subtitle: ""
date: 2025-06-22T12:03:24+08:00
draft: false
author: "June"
authorLink: "https://github.com/zjzjy"
description: ""
images: []
resources:
- name: "featured-image"
  src: "featured-image.jpg"

tags: ["Agent","Agentic RAG"]
categories: ["Agent"]
lightgallery: true

hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: ""

license: '<a rel="license external nofollow noopener noreffer" href="https://creativecommons.org/licenses/by-nc/4.0/" target="_blank">CC BY-NC 4.0</a>'
toc:
  auto: false
---
本文根据[Hugging Face上的Agent课程](https://huggingface.co/learn/agents-course/unit3/agentic-rag/introduction)编写而成，包括        。
在本章节，我们将使用 Agentic RAG 创建一个工具来帮助主持晚会的友好经纪人 Alfred，该工具可用于回答有关晚会嘉宾的问题。
# 难忘的盛会
你决定举办一场本世纪最奢华、最奢华的派对。 这意味着丰盛的宴席、迷人的舞者、知名 DJ、精致的饮品、令人叹为观止的烟火表演等等。我们委托管家Alfred来全权举办这个盛会。为此，他需要掌握派对的所有信息，包括菜单、宾客、日程安排、天气预报等等！不仅如此，他还需要确保聚会取得成功，因此他需要能够在聚会期间回答有关聚会的任何问题 ，同时处理可能出现的意外情况。他无法独自完成这项工作，所以我们需要确保阿尔弗雷德能够获得他所需的所有信息和工具。  
首先，我们给他列一份联欢晚会的硬性要求清单：在文艺复兴时期，受过良好教育的人需要具备三个主要特质：对体育、文化和科学知识的深厚造诣。因此，我们需要确保用我们的知识给宾客留下深刻印象，为他们打造一场真正难忘的盛会。然而，为了避免冲突， 在盛会上应该避免讨论政治和宗教等话题。 盛会需要充满乐趣，避免与信仰和理想相关的冲突。按照礼仪， 一位好的主人应该了解宾客的背景 ，包括他们的兴趣和事业。一位好的主人也会与宾客们闲聊八卦，分享他们的故事。最后，我们需要确保自己掌握一些天气常识 ，以便能够持续获得实时更新，确保在最佳时机燃放烟花，并以一声巨响结束庆典！🎆
# 创建工具
首先，我们将创建一个 RAG 工具，用于检索受邀者的最新详细信息。接下来，我们将开发用于网页搜索、天气更新和 Hugging Face Hub 模型下载统计的工具。
## 为来宾创建 RAG 工具
我们将在[HF Space](https://huggingface.co/spaces/agents-course/Unit_3_Agentic_RAG)开发我们的Agent。
- tools.py：为Agent提供辅助工具。
- retriever.py：实现检索功能，支持知识访问。
- app.py：将所有组件集成到功能齐全的agent中。


使用的[dataset](https://huggingface.co/datasets/agents-course/unit3-invitees)，每个访客包含以下字段：
- Name: 客人的全名
- Relation: 客人与主人的关系
- Description：关于客人的简短传记或有趣的事实
- Email Address：发送邀请或后续活动的联系信息

### Step1: 加载并准备数据集
我们提供了三种不同 Agent 库的实现方式，你可以展开下面的折叠框查看各自的代码。
{{< admonition type=note title="smolagents" open=false >}}
我们将使用 Hugging Face datasets 集库来加载数据集并将其转换为来自 langchain.docstore.document 模块的 Document 对象列表。
```python
import datasets
from langchain.docstore.document import Document

# Load the dataset
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# Convert dataset entries into document objects
docs = [
  Document(
    page_content = "\n".join([
      f"Name: {guest['name']}",
      f"Relation: {guest['relation']}",
      f"Description: {guest['description']}",
      f"Email: {guest['email']}"
    ]),
    metadata={"name": guest["name"]}
  )
  for guest in guest_dataset
]
```
{{< /admonition >}}

{{< admonition type=note title="llama-index" open=false >}}
我们将使用 Hugging Face datasets 集库来加载数据集并将其转换为来自 llama_index.core.schema 模块的 Document 对象列表。
```python
import datasets
from llama_index.core.schema import Document

# Load the dataset
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# Convert dataset entries into Document objects
docs = [
    Document(
        text="\n".join([
            f"Name: {guest_dataset['name'][i]}",
            f"Relation: {guest_dataset['relation'][i]}",
            f"Description: {guest_dataset['description'][i]}",
            f"Email: {guest_dataset['email'][i]}"
        ]),
        metadata={"name": guest_dataset['name'][i]}
    )
    for i in range(len(guest_dataset))
]
```
{{< /admonition >}}

{{< admonition type=note title="langgraph" open=false >}}
我们将使用 Hugging Face datasets 集库来加载数据集并将其转换为来自 langchain.docstore.document 模块的 Document 对象列表。
```python
import datasets
from langchain.docstore.document import Document

# Load the dataset
guest_dataset = datasets.load_dataset("agents-course/unit3-invitees", split="train")

# Convert dataset entries into document objects
docs = [
  Document(
    page_content = "\n".join([
      f"Name: {guest['name']}",
      f"Relation: {guest['relation']}",
      f"Description: {guest['description']}",
      f"Email: {guest['email']}"
    ]),
    metadata={"name": guest["name"]}
  )
  for guest in guest_dataset
]
```
{{< /admonition >}}

在上面的代码中，我们：加载数据集，将每个客人条目转换为具有格式化内容的 Document 对象，将 Document 对象存储在列表中。

### Step2: 创建检索工具
{{< admonition type=note title="smolagents" open=false >}}
我们将使用 langchain_community.retrievers 模块中的 BM25Retriever 来创建检索工具。BM25是相关性搜索，如果要更高级的语义搜索，可以考虑embedding检索器，例如[sentence-transformers ](https://www.sbert.net/)。
```python
from solagents import Tool
from langchain_community.retrievers import BM25Retriever

class GuestInfoRetrieverTool(Tool):
  # 工具的元数据描述
  name = "guest_info_retriever"
  description = "Retrieves detailed information about gala guests based on their name or relation."
  inputs = {
    "query": {
      "type": "string",
      "description": "The name or relation of the guest you want information about."
    }
  }
  output_type = "string"

  def __init__(self, docs):
    self.is_initialized = False
    self.retriever = BM25Retriever.from_document(docs)

  def forward(self, query: str):
    results = self.retriever.get_relevant_documents(query)
    if results:
      return "\n\n".join([doc.page_content for doc in results[:3]])
    else:
      return "No matching guest information found."

# Initialize the tool
guest_info_tool = GuestInfoRetrieverTool(docs)
```
{{< /admonition >}}

{{< admonition type=note title="llama-index" open=false >}}
```python
from llama_index.core.tools import FunctionTool
from llama_index.retrievers.bm25 import BM25Retriever

bm25_retriever = BM25Retriever.from_defaults(nodes = docs)

def get_guest_info_retriever(query: str) -> str:
  """Retrieves detailed information about gala guests based on their name or relation."""
  results = bm25_retriever(query)
  if results:
        return "\n\n".join([doc.text for doc in results[:3]])
  else:
        return "No matching guest information found."

# Initialize the tool
guest_info_tool = FunctionTool.from_defaults(get_guest_info_retriever) 
```
{{< /admonition >}}

{{< admonition type=note title="langgraph" open=false >}}
```python
from langchain_community.retrievers import BM25Retriever
from langchain.tools import Tool

bm25_retriever = BM25Retriever.from_documents(docs)

def extract_text(query: str) -> str:
    """Retrieves detailed information about gala guests based on their name or relation."""
    results = bm25_retriever.invoke(query)
    if results:
        return "\n\n".join([doc.page_content for doc in results[:3]])
    else:
        return "No matching guest information found."

guest_info_tool = Tool(
    name="guest_info_retriever",
    func=extract_text,
    description="Retrieves detailed information about gala guests based on their name or relation."
)
```
{{< /admonition >}}

### Step3：将工具与Alfred集成
最后，让我们通过创建代理并为其配备自定义工具来将所有内容整合在一起：
{{< admonition type=note title="smolagents" open=false >}}
```python
from smolagents import CodeAgent, InferenceClientModel

# Initialize the Hugging Face model
model = InferenceClientModel()

# Create Alfred, our gala agent, with the guest info tool
alfred = CodeAgent(tools=[guest_info_tool], model=model)

# Example query Alfred might receive during the gala
response = alfred.run("Tell me about our guest named 'Lady Ada Lovelace'.")

print("🎩 Alfred's Response:")
print(response)
#🎩 Alfred's Response:
#Based on the information I retrieved, Lady Ada Lovelace is an esteemed mathematician and friend. She is renowned for her pioneering work in mathematics and computing, often celebrated as the first computer programmer due to her work on Charles Babbage's Analytical Engine. Her email address is ada.lovelace@example.com.
```
{{< /admonition >}}

{{< admonition type=note title="llama-index" open=false >}}
```python
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

# Initialize the Hugging Face model
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# Create Alfred, our gala agent, with the guest info tool
alfred = AgentWorkflow.from_tools_or_functions(
    [guest_info_tool],
    llm=llm,
)

# Example query Alfred might receive during the gala
response = await alfred.run("Tell me about our guest named 'Lady Ada Lovelace'.")

print("🎩 Alfred's Response:")
print(response)
#🎩 Alfred's Response:
#Lady Ada Lovelace is an esteemed mathematician and friend, renowned for her pioneering work in mathematics and computing. She is celebrated as the first computer programmer due to her work on Charles Babbage's Analytical Engine. Her email is ada.lovelace@example.com.
```
{{< /admonition >}}

{{< admonition type=note title="langgraph" open=false >}}
```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Generate the chat interface, including the tools
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

chat = ChatHuggingFace(llm=llm, verbose=True)
tools = [guest_info_tool]
chat_with_tools = chat.bind_tools(tools)

# Generate the AgentState and Agent graph
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
    return {
        "messages": [chat_with_tools.invoke(state["messages"])],
    }

## The graph
builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()

messages = [HumanMessage(content="Tell me about our guest named 'Lady Ada Lovelace'.")]
response = alfred.invoke({"messages": messages})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)
#🎩 Alfred's Response:
#Lady Ada Lovelace is an esteemed mathematician and pioneer in computing, often celebrated as the first computer programmer due to her work on Charles Babbage's Analytical Engine.
```
{{< /admonition >}}

最后一步发生了什么：

我们使用 HuggingFaceEndpoint 类初始化 HuggingFace 模型。我们还生成了一个聊天界面并附加了一些工具。

我们将代理（Alfred）创建为 StateGraph ，它使用边组合 2 个节点（ assistant 、 tools ）

我们要求阿尔弗雷德检索有关一位名叫“Lady Ada Lovelace”的客人的信息。


现在 Alfred 可以检索客人信息，请考虑如何增强此系统：
- 改进检索器以使用更复杂的算法，例如句子转换器
- 实现对话记忆 ，以便 Alfred 记住之前的互动
- 结合网络搜索获取陌生客人的最新信息
- 整合多个索引 ，从经过验证的来源获取更完整的信息。
### 小结
关于solagents,llama-index, langgraph 小TIPs:
- 数据加载这块：smolagent和langgraph共用`from langchain.docstore.document import Document`，lamma-index用`from llama_index.core.schema import Document`加载document模块转化为Document object。
- 创建检索工具：
  - 检索工具导入：Smolagent和langgraph使用langchain的BM25检索工具；llama-index使用`from llama_index.retrievers.bm25 import BM25Retriever`
  - Smolagent的Tool库，使用方法：定义一个工具类继承自`Tool`，添加工具的元数据描述(name, description, inputs)，定义`forward`方法。
  - llama-index的`from llama_index.core.tools import FunctionTool`，直接定义python函数即可（注意要添加函数描述），最后`FunctionTool.from_defaults(get_guest_info_retriever) `初始化工具
  - langgraph的`from langchain.tools import Tool`，定义python函数（注意要添加函数描述），初始化工具步骤需要手动描述(name, func,description), like
  ```python
  guest_info_tool = Tool(
    name="guest_info_retriever",
    func=extract_text,
    description="Retrieves detailed information about gala guests based on their name or relation.")
   ```
- 工具集成：
  - Smolagent：`from smolagents import CodeAgent, InferenceClientModel`，初始化model，直接使用CodeAgent即可，`alfred = CodeAgent(tools=[guest_info_tool], model=model)`。
  - llama-index：`from llama_index.core.agent.workflow import AgentWorkflow`，`from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI`，初始化定义好llm，直接用Agentflow即可`alfred = AgentWorkflow.from_tools_or_functions([guest_info_tool],llm=llm,)`。
  - langgraph：是个大工程。需要先初始化llm，工具列表，将llm与工具表绑定。初始化graph，定义node，定义edge。
## 为您的Agent构建和集成工具
在本节中，我们将授予 Alfred 访问网络的权限，使他能够查找最新新闻和全球动态。此外，他还将能够访问天气数据和 Hugging Face 中心模型的下载统计数据，以便他就热门话题进行相关对话。
### Give Your Agent Access to the Web
我们需要确保Alfred德能够获取有关世界的最新新闻和信息。  
让我们从为 Alfred 创建一个网络搜索工具开始吧！
{{< admonition type=note title="solagents" open=false >}}
```python
from smolagents import DuckDuckGoSearchTool

search_tool = DuckDuckGoSearchTool()

# Example usage
results = search_tool("Who's the current President of France?")
print(resultts)
# The current President of France in Emmanuel Macron.
```
{{< /admonition >}}

{{< admonition type=note title="llama-index" open=false >}}
```python
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
from llama_index.core.tools import FunctionTool

# Initialize the DuckDuckGo search  tool
tool_spec = DuckDuckGoSearchToolSpec()

search_tool = FunctionTool.from_defaults(tool_spec.duckduckgo_full_search)

# Example usage
results = search_tool("Who's the current President of France?")
print(resultts.raw_output[-1]['body'])
# The President of the French Republic is the head of state of France. The current President is Emmanuel Macron since 14 May 2017 defeating Marine Le Pen in the second round of the presidential election on 7 May 2017. List of French presidents (Fifth Republic) N° Portrait Name ...
```
{{< /admonition >}}

{{< admonition type=note title="langgraph" open=false >}}
```python
from langchain.community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

# Example usage
results = search_tool.invoke("Who's the current President of France?")
print(resultts)
# Emmanuel Macron (born December 21, 1977, Amiens, France) is a French banker and politician who was elected president of France in 2017...
```
{{< /admonition >}}

### 创建自定义工具来获取天气信息以安排烟花表演
完美的庆典应该是在晴朗的天空下燃放烟花，我们需要确保烟花不会因为恶劣的天气而取消。让我们创建一个自定义工具，可用于调用外部天气 API 并获取给定位置的天气信息。为了简单起见，我们在本例中使用了一一个虚拟天气API，如果您想用真实的天气API，可以使用OpenWeatherMap API等。
{{< admonition type=note title="smolagents" open=false>}}
```python
from smolagents import Tool
import random

class WeatherInfoTool(Tool):
  name = "weather_info"
  description = "Fetches dummy weather information for a given location."
  inputs = {
    "location" : {
      "type" : "string",
      "description": "The location to get weather information for."
    }
  }
  output_type = "string"

  def forward(self, location:str):
    # Dummy weather data
    weather_conditions = [
      {"condition": "Rainy", "temp_c": 15},
      {"condition": "Clear", "temp_c": 25},
      {"condition": "Windy", "temp_c": 20}
    ]
    # Randomly select a weather condition
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

# Initialize the tool
weather_info_tool = WeatherInfoTool()
 ```
{{< /admonition >}}

{{< admonition type=note title="llama-index" open=false>}}
```python
import random
from llama_index.core.tools import FunctionTool

def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # Dummy weather data
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20}
    ]
    # Randomly select a weather condition
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

# Initialize the tool
weather_info_tool = FunctionTool.from_defaults(get_weather_info)
```
{{< /admonition >}}

{{< admonition type=note title="langgraph" open=false>}}
```python
from langchain.tools import Tool
import random

def get_weather_info(location: str) -> str:
    """Fetches dummy weather information for a given location."""
    # Dummy weather data
    weather_conditions = [
        {"condition": "Rainy", "temp_c": 15},
        {"condition": "Clear", "temp_c": 25},
        {"condition": "Windy", "temp_c": 20}
    ]
    # Randomly select a weather condition
    data = random.choice(weather_conditions)
    return f"Weather in {location}: {data['condition']}, {data['temp_c']}°C"

# Initialize the tool
weather_info_tool = Tool(
    name="get_weather_info",
    func=get_weather_info,
    description="Fetches dummy weather information for a given location."
)
```
{{< /admonition >}}

### 为AI Builders创建 Hub Stats Tool
出席此次盛会的都是 AI 开发者的精英。Alfred 希望通过讨论他们最受欢迎的模型、数据集和空间来给他们留下深刻印象。我们将创建一个工具，根据用户名从 Hugging Face Hub 获取模型统计数据。
{{< admonition type=note title="smolagents" open=false>}}
```python
from solagents import Tool
from huggingface_hub import list_models

class HubStatsTool(Tool):
  name = "hub_stats"
  description = "Fetches the most downloaded model from a specific author on the Hugging Face Hub."
  inputs = {
    "author":{
      "type": "string",
      "description": "The username of the model author/organization to find models from."
    }
  }
  output_type = "string"

  def forward(self, author: str):
    try:
      # List Models from the specified author, sorted by downloads
      models = list(list_models(author = author, sort = "sdownloads", direction=-1, limit=1))

      if models:
        model = models[0]
        return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
      else:
        return f"Nomodels found for author {author}"
    except Exception as e:
      return f"Error fetching models for {author}: {str(e)}"
# Initialize the tool
hub_stats_tool = HubStatsTool()

# Example usage
print(hub_stats_tool("facebook")) # Example: Get the most downloaded model by Facebook
# The most downloaded model by facebook is facebook/esmfold_v1 with 12,544,550 downloads.
```
{{< /adminiton >}}

{{< admonition  type=note title="llama-index" open=false>}}
```python
import random
from llama_index.core.tools import FunctionTool
from huggingface_hub import list_models

def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        # List models from the specified author, sorted by downloads
        models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))

        if models:
            model = models[0]
            return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
        else:
            return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"

# Initialize the tool
hub_stats_tool = FunctionTool.from_defaults(get_hub_stats)

# Example usage
print(hub_stats_tool("facebook")) # Example: Get the most downloaded model by Facebook
```
{{< /adminiton >}}
{{< admonition type=note title="langgraph" open=false>}}
```python
from langchain.tools import Tool
from huggingface_hub import list_models

def get_hub_stats(author: str) -> str:
    """Fetches the most downloaded model from a specific author on the Hugging Face Hub."""
    try:
        # List models from the specified author, sorted by downloads
        models = list(list_models(author=author, sort="downloads", direction=-1, limit=1))

        if models:
            model = models[0]
            return f"The most downloaded model by {author} is {model.id} with {model.downloads:,} downloads."
        else:
            return f"No models found for author {author}."
    except Exception as e:
        return f"Error fetching models for {author}: {str(e)}"

# Initialize the tool
hub_stats_tool = Tool(
    name="get_hub_stats",
    func=get_hub_stats,
    description="Fetches the most downloaded model from a specific author on the Hugging Face Hub."
)

# Example usage
print(hub_stats_tool("facebook")) # Example: Get the most downloaded model by Facebook
```
{{< /adminition >}}
### 工具集成
现在我们已经拥有了所有的工具，让我们将它们集成到 Alfred 的代理中：
{{< abminition type=note title="smolagents" open=false>}}
```python
from smolagents import CodeAgent, InferenceClientModel

model = InferenceModel()

alfred = CodeAgent(
  tools = [search_tool, weather_info_tool, hub_stats_tool], 
  model = model
)
# Example query Alfred might receive during the gala
response = alfred.run("What is Facebook and what's their most popular model?")

print("🎩 Alfred's Response:")
print(response)
```
{{< /admintion >}}
{{< abminition type=note title="llama-index" open=false>}}
```python
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

# Initialize the Hugging Face model
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")
# Create Alfred with all the tools
alfred = AgentWorkflow.from_tools_or_functions(
    [search_tool, weather_info_tool, hub_stats_tool],
    llm=llm
)

# Example query Alfred might receive during the gala
response = await alfred.run("What is Facebook and what's their most popular model?")

print("🎩 Alfred's Response:")
print(response)
```
{{< /admintion >}}
{{< abminition type=note title="langgraph" open=false>}}
```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebulit import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# Generate the chat interface, including the tools
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

chat = ChatHuggingFace(llm = llm, verbose = True)
tools = [search_tool, weather_info_tool, hub_stats_tool]
chat_with_tools = chat.bind_tools(tools)

# Generate the AgentState and Agent graph
class AgentState(TypedDict):
  messages: Annotated[list[AnyMessage], add_messages]

def assistant(state: AgentState):
  return {
    "messages": [chat_with_tools.invoke(state["messages"])],
  }

# The graph
builder = StateGraph(AgentState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges: these determine how control flow moves
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message requires a tool, route to tools
    # Otherwise, provide a direct response
    tools_condition,
)
builder.add_edge("tools", "assistant")
alfred = builder.compile()

messages = [HumanMessages(content="Who is Facebook and what's their most popular model?")]
response = alfred.invoke({"messages": messages})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)
```
{{< /admintion >}}