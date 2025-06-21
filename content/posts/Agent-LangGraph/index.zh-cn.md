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


# LangGraph的构建模块

# 构建一个邮件分类助手吧

# 构建一个文件分析agent吧
