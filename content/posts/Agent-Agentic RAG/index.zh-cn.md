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


我们需要做的步骤：
1. 加载并准备数据集
  
2. 创建检索工具
3. 将工具与Alfred集成

我们提供了三种不同 Agent 库的实现方式，你可以展开下面的折叠框查看各自的代码。

{{< admonition type=note title="使用 smolagents 实现" open=false >}}
```python
# smol-agent code placeholder
print("This is code for smolagents")
```
{{< /admonition >}}

{{< admonition type=note title="使用 llama-index 实现" open=false >}}
```python
# llama-index code placeholder
print("This is code for llama-index")
```
{{< /admonition >}}

{{< admonition type=note title="使用 langgraph 实现" open=false >}}
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
```
{{< /admonition >}}

