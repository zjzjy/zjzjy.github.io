---
weight: 1
title: "LangGraph - Introduction"
subtitle: ""
date: 2025-06-21T10:21:11+08:00
draft: false
author: "June"
authorLink: "https://github.com/zjzjy"
description: ""
images: []
resources:
- name: "featured-image"
  src: "featured-image.jpg"

tags: ['Agent','LangGraph']
categories: ['Agent']
lightgallery: true

hiddenFromHomePage: false
hiddenFromSearch: false

featuredImage: ""
featuredImagePreview: ""

license: '<a rel="license external nofollow noopener noreffer" href="https://creativecommons.org/licenses/by-nc/4.0/" target="_blank">CC BY-NC 4.0</a>'
toc:
  auto: false
---
本文根据[Hugging Face上的Agent课程](https://huggingface.co/learn/agents-course/unit2/langgraph/introduction)编写而成，包括        。
在本章节您将学习如何使用 [LangGraph](https://github.com/langchain-ai/langgraph) 框架构建应用程序，该框架旨在帮助您构建和协调复杂的 LLM 工作流程。LangGraph 是一个框架，它通过为您提供代理流程的控制工具，允许您构建可用于生产的应用程序。
相关资源：
- [LangGraph 代理](https://langchain-ai.github.io/langgraph/) - LangGraph 代理示例
- [LangChain academy](https://academy.langchain.com/courses/intro-to-langgraph) - Full course on LangGraph from LangChain
# 什么是LangGraph，什么时候使用它？
LangGraph 是 [LangChain](https://www.langchain.com/) 开发的用于管理集成 LLM 的应用程序的控制流的框架。  
**那么，LangGraph与LangChain有什么不同？**LangChain 提供了一个标准接口，用于与模型和其他组件交互，可用于检索、LLM 调用和工具调用。LangChain 中的类可以在 LangGraph 中使用，但并非必须使用。这些包是不同的，可以单独使用，但最终，您在网上找到的所有资源都会同时使用这两个包。  
**什么时候应该使用 LangGraph？**  
当你需要做一个“控制”和“自由”之间的权衡：
- 控制：确保可预测行为并维护。
- 自由：让LLM有更多空间去发挥创造力。  
例如：CodeAgent非常自由，可以在单个操作步骤中调用多个工具，创建自己的工具等等，但这种行为可能让它们比使用JSON的常规代理更难以预测和控制。  

LangGraph则处于另一个极端，当您需要“控制”agent的执行时，就会发挥作用。它为您提供了构建遵循可预测流程的应用程序的工具，同时仍然充分利用 LLM 的强大功能。简而言之，如果您的应用程序涉及**一系列需要以特定方式协调的步骤，并且在每个连接点做出决策**， 那么 LangGraph 可以提供您所需的结构 。

LangGraph 擅长的关键场景包括：
- 需要明确控制流程的多步骤推理过程
- 需要在步骤之间保持状态的应用程序
- 将确定性逻辑与人工智能功能相结合的系统
- 需要人工干预的工作流程
- 具有多个组件协同工作的复杂代理架构
# LangGraph的构建模块
LangGraph 中的应用程序从入口点开始，并且根据执行情况，流程可能会转到一个函数或另一个函数，直到到达结束。
![LangGraph示意图](/assets/images/langgraph_node.png)
## State
State是 LangGraph 的核心概念。它代表了流经应用程序的所有信息。
```python
from typing_extensions import TypeDict

class State(TyprDict):
  graph_state: str
```
状态是用户定义的 ，因此字段应该精心设计以包含决策过程所需的所有数据！💡： 仔细考虑您的应用程序需要在步骤之间跟踪哪些信息。
## Node
Node 时Python函数。每个Node：
- 将状态作为输入
- 执行操作
- 返回状态更新
```python
def node_1(state):
  print("---Node 1----")
  return {"graph_state": state['graph_state']+"I am"}

def node_2(state):
    print("---Node 2---")
    return {"graph_state": state['graph_state'] +" happy!"}

def node_3(state):
    print("---Node 3---")
    return {"graph_state": state['graph_state'] +" sad!"}
```
Node可以包括什么呢？
- LLM 调用 ：生成文本或做出决策
- 工具调用 ：与外部系统交互
- 条件逻辑 ：确定下一步
- 人工干预 ：获取用户输入
整个工作流程所需的一些Node（如 START 和 END）直接存在于 langGraph 中。

## Edge
Edge连接Node并定义图中的可能路径：
```python
import random
from typing import Literal # Literal 类型允许你明确规定变量的具体可选值，这些值可以是字符串、整数、布尔值等不可变类型。
def decide_mood(state) -> Literal["node_2", "node_3"]:
    
    # Often, we will use state to decide on the next node to visit
    user_input = state['graph_state'] 
    
    # Here, let's just do a 50 / 50 split between nodes 2, 3
    if random.random() < 0.5:

        # 50% of the time, we return Node 2
        return "node_2"
    
    # 50% of the time, we return Node 3
    return "node_3"
```
Edges可以是：
- 直接 ：始终从节点 A 到节点 B
- 条件 ：根据当前状态选择下一个节点

## StateGraph
StateGraph 是保存整个代理工作流程的容器：
```python
from IPython.display import Image, display
from langgraph.graph import StaeGraph, START, END

# 创建状态图并添加节点
builder = StateGraph(State)
builder.add_node("node_1", node_1)
builder.add_node("node_2", node_2)
builder.add_node("node_3", node_3)

# 定义节点之间的连接关系（边）
builder.add_edge(START, "node_1")
builder.add_conditional_edges("node_1", decide_mood)
builder.add_edge("node_2", END)
builder.add_edge("node_3", END)

graph = builder.compile()

# View
display(Image(graph.get_graph().draw_mermaid_png()))

# 调用
graph.invoke({"graph_state": "Hi, this is Lance."})
# 输出
#---Node 1---
#---Node 3---
#{'graph_state': 'Hi, this is Lance. I am sad!'}
```
# 构建一个邮件助手吧！
在这一小节，我们会实现Alfred的电子邮件处理系统，他需要执行以下操作：
1. 阅读收到的电子邮件
2. 将其归类为垃圾邮件或合法邮件
3. 起草对合法电子邮件的初步回复
4. 在合法的情况下向韦恩先生发送信息（仅打印）
这是我们将构建的工作流程：
![email workflow](/assets/images/langgraph_email.png)
## 设置环境
```python
pip install langgraph langchain_openai

import os
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
```

## Step 1: Define Our State
使您的State足够全面以跟踪所有重要信息，但避免添加不必要的细节。
```python
class EmailState(TypedDict)
  # The email being processed
  email: Dict[str, Any] # Contains subject, sender, body, etc.

   # Category of the email (inquiry, complaint, etc.)
   email_category: Optional[str]

   # Reason why the email was marked as spam
   spam_reason: Optional[str]

   # Analysis and decisions
   is_spam: Optional[bool]

   # Response generation
   email_draft: Optional[str]

   # Processing metadata
   messages: List[DItc[str, Any]]
```
## Step 2: Define Our Nodes
现在我们创建构成节点的处理函数，想一想我们需要什么？
1. 小助手要读邮件，返回logs: 小助手在处理来自发送者某某关于某某主题的邮件
2. 小助手要判断是不是垃圾邮件，从LLM回答中提取is_spam，reason，category。
3. 小助手处理垃圾邮件
4. 小助手起草回复
5. 小助手回复整个过程
```python
# Initialize LLM
model = ChatOpenAI(temparature=0)

def read_email(state: EmailState):
  """Alfred reads and logs the incoming email"""
  email = state["email"]

  # Here we might do some initial preprocessing
  print(f"Alfred is processing an email from {email['sender']} with subject: {email['subject']}")

  # No state changes needed here
  return {}

def classify_email(state: EmailState):
  """Alfred uses an LLM to determine if the email is spam or legitimate"""
  email = state["email"]

  # prepare our prompt for the LLM
  prompt = f"""
    As Alfred the butler, analyze this email and determine if it is spam or legitimate.
    
    Email:
    From: {email['sender']}
    Subject: {email['subject']}
    Body: {email['body']}
    
    First, determine if this email is spam. If it is spam, explain why.
    If it is legitimate, categorize it (inquiry, complaint, thank you, etc.).
    """

    # call the LLM
    messages = [HumanMessage(content=prompt)]
    response = model.invoke(messages)

    # Simple logic to parse the response (in a real app, you'd want more robust parsing)
    
```
# 构建一个文件分析agent吧！
