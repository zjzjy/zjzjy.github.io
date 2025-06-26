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
本文根据[Hugging Face上的Agent课程](https://huggingface.co/learn/agents-course/unit3/agentic-rag/introduction)编写而成。
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
{{< admonition type=note title=smolagents open=false >}}
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

{{< admonition type=note title=llama-index open=false >}}
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

{{< admonition type=note title=langgraph open=false >}}
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
{{< admonition type=note title=smolagents open=false >}}
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

{{< admonition type=note title=llama-index open=false >}}
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

{{< admonition type=note title=langgraph open=false >}}
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
{{< admonition type=note title=smolagents open=false >}}
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

{{< admonition type=note title=llama-index open=false >}}
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

{{< admonition type=note title=langgraph open=false >}}
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
{{< admonition type=note title=solagents open=false >}}
```python
from smolagents import DuckDuckGoSearchTool

search_tool = DuckDuckGoSearchTool()

# Example usage
results = search_tool("Who's the current President of France?")
print(resultts)
# The current President of France in Emmanuel Macron.
```
{{< /admonition >}}

{{< admonition type=note title=llama-index open=false >}}
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

{{< admonition note "langgraph" false>}}
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
{{< admonition note "smolagents" false>}}
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

{{< admonition note "llama-index" false>}}
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

{{< admonition note "langgraph" false>}}
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
{{< admonition note "smolagents" false>}}
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
{{< /admonition >}}

{{< admonition note "llama-index" false>}}
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
{{< /admonition >}}
{{< admonition note "langgraph" false>}}
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
{{< /admonition >}}
### 工具集成
现在我们已经拥有了所有的工具，让我们将它们集成到 Alfred 的代理中：
{{< admonition note "smolagents" false>}}
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
{{< /admonition >}}
{{< admonition note "llama-index" false>}}
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
{{< /admonition >}}
{{< admonition note "langgraph" false>}}
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
{{< /admonition >}}

## Creating Your Gala Agent  
现在我们已经为 Alfred 构建了所有必要的组件，现在是时候将所有组件整合成一个完整的代理，以帮助我们举办奢华的盛会。  
在本节中，我们将把客人信息检索、网络搜索、天气信息和 Hub 统计工具组合成一个强大的代理。  
我们在之前已经实现了tools.py和retriever.py，接下来要导入它们。
{{< admonition note "solagents" false>}}
```python
# Import necessary libraries
import random
from smolagents import CodeAgent, InferenceClientModel

# Import our custom tools from their modules
from tools import DuckDuckGoSearchTool, WeatherInfoTool, HubStatsTool
from retriever import load_guest_dataset

# Initialize the Hugging Face model
model = InferenceClientModel()

# Initialize the web search tool
search_tool = DuckDuckGoSearchTool()

# Initialize the weather tool
weather_info_tool = WeatherInfoTool()

# Initialize the Hub stats tool
hub_stats_tool = HubStatsTool()

# Load the guest dataset and initialize the guest info tool
guest_info_tool = load_guest_dataset()

# Create Alfred with all the tools
alfred = CodeAgent(
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool], 
    model=model,
    add_base_tools=True,  # Add any additional base tools
    planning_interval=3   # Enable planning every 3 steps
)
```
{{< /admonition >}}
{{< admonition note "llama-index" false>}}
```python
# Import necessary libraries
from llama_index.core.agent.workflow import AgentWorkflow
from llama_index.llms.huggingface_api import HuggingFaceInferenceAPI

from tools import search_tool, weather_info_tool, hub_stats_tool
from retriever import guest_info_tool

# Initialize the Hugging Face model
llm = HuggingFaceInferenceAPI(model_name="Qwen/Qwen2.5-Coder-32B-Instruct")

# Create Alfred with all the tools
alfred = AgentWorkflow.from_tools_or_functions(
    [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool],
    llm=llm,
)
```
{{< /admonition >}}
{{< admonition note "langgraph" false>}}
```python
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

from tools import DuckDuckGoSearchRun, weather_info_tool, hub_stats_tool
from retriever import guest_info_tool

# Initialize the web search tool
search_tool = DuckDuckGoSearchRun()

# Generate the chat interface, including the tools
llm = HuggingFaceEndpoint(
    repo_id="Qwen/Qwen2.5-Coder-32B-Instruct",
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN,
)

chat = ChatHuggingFace(llm=llm, verbose=True)
tools = [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool]
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
```
{{< /admonition >}}
# 应用示范
我们将使用刚刚创建的Alfred Agent完成三个工作：
1. 查找客人信息
2. 检查烟花天气
3. 给AI Builder 客人留下深刻印象
4. 使用多个工具与尼古拉斯博士进行对话
{{< admonition note "smolagents" false>}}
```python 
# 1.查找客人信息
query = "Tell me about 'Lady Ada Lovelace'"
response = alfred.run(query)
print("🎩 Alfred's Response:")
print(response)
# 2.检查烟花天气
query = "What's the weather like in Paris tonight? Will it be suitable for our fireworks display?"
response = alfred.run(query)

print("🎩 Alfred's Response:")
print(response)
# 3. 给AI Builder 客人留下深刻印象
query = "One of our guests is from Qwen. What can you tell me about their most popular model?"
response = alfred.run(query)

print("🎩 Alfred's Response:")
print(response)
# 4. 使用多个工具与尼古拉斯博士进行对话
query = "I need to speak with Dr. Nikola Tesla about recent advancements in wireless energy. Can you help me prepare for this conversation?"
response = alfred.run(query)

print("🎩 Alfred's Response:")
print(response)
```
ALfred的回答：
```js
🎩 Alfred's Response:
I've gathered information to help you prepare for your conversation with Dr. Nikola Tesla.

Guest Information:
Name: Dr. Nikola Tesla
Relation: old friend from university days
Description: Dr. Nikola Tesla is an old friend from your university days. He's recently patented a new wireless energy transmission system and would be delighted to discuss it with you. Just remember he's passionate about pigeons, so that might make for good small talk.
Email: nikola.tesla@gmail.com

Recent Advancements in Wireless Energy:
Based on my web search, here are some recent developments in wireless energy transmission:
1. Researchers have made progress in long-range wireless power transmission using focused electromagnetic waves
2. Several companies are developing resonant inductive coupling technologies for consumer electronics
3. There are new applications in electric vehicle charging without physical connections

Conversation Starters:
1. "I'd love to hear about your new patent on wireless energy transmission. How does it compare to your original concepts from our university days?"
2. "Have you seen the recent developments in resonant inductive coupling for consumer electronics? What do you think of their approach?"
3. "How are your pigeons doing? I remember your fascination with them."

This should give you plenty to discuss with Dr. Tesla while demonstrating your knowledge of his interests and recent developments in his field.
```
{{< /admonition >}}

{{< admonition note "llama-index" false>}}
```python 
# 1.查找客人信息
query = "Tell me about 'Lady Ada Lovelace'"
response = await alfred.run(query)
print("🎩 Alfred's Response:")
print(response.response.blocks[0].text)
# 2.检查烟花天气
query = "What's the weather like in Paris tonight? Will it be suitable for our fireworks display?"
response = await alfred.run(query)

print("🎩 Alfred's Response:")
print(response)
# 3. 给AI Builder 客人留下深刻印象
query = "One of our guests is from Google. What can you tell me about their most popular model?"
response = await alfred.run(query)

print("🎩 Alfred's Response:")
print(response)
# 4. 使用多个工具与尼古拉斯博士进行对话
query = "I need to speak with Dr. Nikola Tesla about recent advancements in wireless energy. Can you help me prepare for this conversation?"
response = await alfred.run(query)

print("🎩 Alfred's Response:")
print(response)
```
ALfred的回答：
```json
🎩 Alfred's Response:
Here are some recent advancements in wireless energy that you might find useful for your conversation with Dr. Nikola Tesla:

1. **Advancements and Challenges in Wireless Power Transfer**: This article discusses the evolution of wireless power transfer (WPT) from conventional wired methods to modern applications, including solar space power stations. It highlights the initial focus on microwave technology and the current demand for WPT due to the rise of electric devices.

2. **Recent Advances in Wireless Energy Transfer Technologies for Body-Interfaced Electronics**: This article explores wireless energy transfer (WET) as a solution for powering body-interfaced electronics without the need for batteries or lead wires. It discusses the advantages and potential applications of WET in this context.

3. **Wireless Power Transfer and Energy Harvesting: Current Status and Future Trends**: This article provides an overview of recent advances in wireless power supply methods, including energy harvesting and wireless power transfer. It presents several promising applications and discusses future trends in the field.

4. **Wireless Power Transfer: Applications, Challenges, Barriers, and the
```
{{< /admonition >}}

{{< admonition note "langgraph" false>}}
```python 
# 1.查找客人信息
response = alfred.invoke({"messages": "Tell me about 'Lady Ada Lovelace'"})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)
# 2.检查烟花天气
response = alfred.invoke({"messages": "What's the weather like in Paris tonight? Will it be suitable for our fireworks display?"})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)
# 3. 给AI Builder 客人留下深刻印象
response = alfred.invoke({"messages": "One of our guests is from Qwen. What can you tell me about their most popular model?"})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)
# 4. 使用多个工具与尼古拉斯博士进行对话
response = alfred.invoke({"messages":"I need to speak with 'Dr. Nikola Tesla' about recent advancements in wireless energy. Can you help me prepare for this conversation?"})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)
```
ALfred的回答：
```json
Based on the provided information, here are key points to prepare for the conversation with 'Dr. Nikola Tesla' about recent advancements in wireless energy:\n1. **Wireless Power Transmission (WPT):** Discuss how WPT revolutionizes energy transfer by eliminating the need for cords and leveraging mechanisms like inductive and resonant coupling.\n2. **Advancements in Wireless Charging:** Highlight improvements in efficiency, faster charging speeds, and the rise of Qi/Qi2 certified wireless charging solutions.\n3. **5G-Advanced Innovations and NearLink Wireless Protocol:** Mention these as developments that enhance speed, security, and efficiency in wireless networks, which can support advanced wireless energy technologies.\n4. **AI and ML at the Edge:** Talk about how AI and machine learning will rely on wireless networks to bring intelligence to the edge, enhancing automation and intelligence in smart homes and buildings.\n5. **Matter, Thread, and Security Advancements:** Discuss these as key innovations that drive connectivity, efficiency, and security in IoT devices and systems.\n6. **Breakthroughs in Wireless Charging Technology:** Include any recent breakthroughs or studies, such as the one from Incheon National University, to substantiate the advancements in wireless charging.
```
{{< /admonition >}}
# 高级功能：记忆！
为了让 Alfred 在晚会上提供更多帮助，我们可以启用对话记忆功能，以便他记住之前的互动：
{{< admonition note "smolagents" false>}}
```python
# Create Alfred with conversation memory
alfred_with_memory = CodeAgent(
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool, search_tool], 
    model=model,
    add_base_tools=True,
    planning_interval=3 # 代理每执行 3 个工具调用 后，会基于当前记忆重新规划后续步骤。
)

# First interaction
response1 = alfred_with_memory.run("Tell me about Lady Ada Lovelace.")
print("🎩 Alfred's First Response:")
print(response1)

# Second interaction (referencing the first)
response2 = alfred_with_memory.run("What projects is she currently working on?", reset=False)
print("🎩 Alfred's Second Response:")
print(response2)
```
{{< /admonition >}}

{{< admonition note "llama-index" false>}}
```python
from llama_index.core.workflow import Context

alfred = AgentWorkFlow.from_tools_or_functions(
  [guest_info_tool, search_tool, weather_info_tool, hub_stats_tool],
    llm=llm
)

# Remembering state
ctx = Context(alfred)

# First interaction
response1 = await alfred.run("Tell me about Lady Ada Lovelace.", ctx=ctx)
print("🎩 Alfred's First Response:")
print(response1)

# Second interaction (referencing the first)
response2 = await alfred.run("What projects is she currently working on?", ctx=ctx)
print("🎩 Alfred's Second Response:")
print(response2)
```
{{< /admonition >}}

{{< admonition note "langgraph" false>}}
显式传递消息
```python
# First interaction
response = alfred.invoke({"messages": [HumanMessage(content="Tell me about 'Lady Ada Lovelace'. What's her background and how is she related to me?")]})


print("🎩 Alfred's Response:")
print(response['messages'][-1].content)
print()

# Second interaction (referencing the first)
response = alfred.invoke({"messages": response["messages"] + [HumanMessage(content="What projects is she currently working on?")]})

print("🎩 Alfred's Response:")
print(response['messages'][-1].content)
```
{{< /admonition >}}
总结一下：
- smolagents：内存不会在不同的执行运行中保留，您必须使用 reset=False 明确声明它。
- LlamaIndex：需要在运行中明确添加用于内存管理的上下文对象。
- LangGraph：提供检索以前的消息或使用专用 [MemorySaver ](https://langchain-ai.github.io/langgraph/concepts/why-langgraph/#part-3-adding-memory-to-the-chatbot)组件的选项。


完结撒花~