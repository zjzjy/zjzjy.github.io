# Agentic RAG - Usecase

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
- 整合多个索引 ，从经过验证的来源获取更完整的信息
