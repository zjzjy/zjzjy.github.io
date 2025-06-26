# Smolagents - Introduction

本文根据[Hugging Face上的Agent课程](https://huggingface.co/learn/agents-course/unit2/smolagents/introduction)编写而成。
相关资源：
- [smolagents Documentation](https://huggingface.co/docs/smolagents) - Official docs for the smolagents library
- [Building Effective Agents](https://www.anthropic.com/engineering/building-effective-agents) - Research paper on agent architectures
- [Agent Guidelines](https://huggingface.co/docs/smolagents/tutorials/building_good_agents) - Best practices for building reliable agents
- [LangGraph Agents](https://langchain-ai.github.io/langgraph/) - Additional examples of agent implementations
- [Function Calling Guide](https://platform.openai.com/docs/guides/function-calling?api-mode=chat) - Understanding function calling in LLMs
- [RAG Best Practices](https://www.pinecone.io/learn/retrieval-augmented-generation/) - Guide to implementing effective RAG
- [smolagents Blog](https://huggingface.co/blog/smolagents) - Introduction to smolagents and code interactions
# 什么是smolagents，为什么要使用smolagents？
smolagents 是一个 [Hugging Face 库](https://github.com/huggingface/smolagents)。smolagents 是一个简单但功能强大的 AI 代理构建框架。它为LLM提供了与现实世界交互的能力，例如搜索或生成图像。
## 主要优势
+ **简单性**： 最小的代码复杂性和抽象性，使框架易于理解、采用和扩展。
+ **灵活的 LLM 支持**： 通过与 Hugging Face 工具和外部 API 集成，可与任何 LLM 配合使用
+ **代码优先方法**： 对代码代理提供一流的支持，这些代理直接在代码中编写其操作，无需解析并简化工具调用
+ **HF Hub 集成**： 与 Hugging Face Hub 无缝集成，允许使用 Gradio Spaces 作为工具
## 何时适合使用smolagents?
- 您需要一个轻量级且最小的解决方案。
- 您希望快速进行实验而无需进行复杂的配置。
- 您的应用程序逻辑很简单。

以上的情况适合使用smolagents。

## smolagents中的模型集成
smolagents 支持灵活的 LLM 集成，允许您使用任何符合特定条件的可调用模型。该框架提供了几个预定义的类来简化模型连接：
- [TransformersModel](https://huggingface.co/docs/smolagents/main/en/reference/models#smolagents.TransformersModel)： 实现本地 transformers 管道，实现无缝集成。
```python
from smolagents import TransformersModel

model = TransformersModel(model_id="HuggingFaceTB/SmolLM-135M-Instruct")

print(model([{"role": "user", "content": [{"type": "text", "text": "Ok!"}]}], stop_sequences=["great"]))
```
- [InferenceClientModel](https://huggingface.co/docs/smolagents/main/en/reference/models#smolagents.InferenceClientModel) ： 支持通过 [Hugging Face 的基础设施](https://huggingface.co/docs/huggingface_hub/main/en/guides/inference)或通过越来越多的[第三方推理提供商进行无服务器推理调用](https://huggingface.co/docs/huggingface_hub/main/en/guides/inference#supported-providers-and-tasks)。
HfApiModel 封装了 huggingface_hub 的 InferenceClient ，用于执行 LLM。它支持 Hub 上所有可用的推理提供程序 ：Cerebras、Cohere、Fal、Fireworks、HF-Inference、Hyperbolic、Nebius、Novita、Replicate、SambaNova、Together 等。
```python
from smolagents import InferenceClientModel

message = [
  {"role": "user", "content": [{"type": "text", "text": "Hello, how are you?"}]}
]

model = InferenceClientModel(provider = "novita")
print(model(messages))
```
- [LiteLLMModel](https://huggingface.co/docs/smolagents/main/en/reference/models#smolagents.LiteLLMModel) ： 利用 LiteLLM 实现轻量级模型交互。
```python
from solagents import LiteLLMModel

messages = [
  {"role": "user", "content": [{"type": "text", "text": "Hello, how are you?"}]}
]

model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-latest", temperature=0.2, max_tokens=10)
print(model(messages))
```
- [OpenAIServerModel](https://huggingface.co/docs/smolagents/main/en/reference/models#smolagents.OpenAIServerModel) ： 连接到任何提供 OpenAI API 接口的服务。
```python
import os
from smolagents import OpenAIServerModel

model = OpenAIServerModel(
  model_id = "gpt-4o",
  api_base = "https://api.openai.com/v1",
  api_key = os.environ["OPENAI_API_KEY"],
)
```
- [AzureOpenAIServerModel](https://huggingface.co/docs/smolagents/main/en/reference/models#smolagents.AzureOpenAIServerModel) ： 支持与任何 Azure OpenAI 部署集成。
```python
import os

from smolagents import AzureOpenAIServerModel

model = AzureOpenAIServerModel(
    model_id = os.environ.get("AZURE_OPENAI_MODEL"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    api_version=os.environ.get("OPENAI_API_VERSION")    
)
```
# Agent类型 
## CodeAgents
使用代码而不是 JSON 编写操作有几个主要优势：
- 可组合性 ：轻松组合和重用操作
- 对象管理 ：直接处理图像等复杂结构
- 通用性 ：表达任何计算上可能的任务

这对 LLM 来说很自然 ：LLM 训练数据中已经存在高质量的代码。
它是核心构建块。CodeAgent 是一种特殊的 MultiStepAgent ， CodeAgent 将在下面的示例中看到。

### CodeAgent逻辑流程
![how code agent works?](images/codeagent_run.png)
CodeAgent 通过一系列步骤执行操作，将现有变量和知识纳入代理的上下文中，并保存在执行日志中：
1. 系统提示存储在 SystemPromptStep 中，用户查询记录在 TaskStep 中。
2. 然后，执行以下 while 循环：
   1. agent.write_memory_to_messages() 将代理的日志写入 LLM 可读的chat messages中。
   2. 这些消息被发送到一个 Model ，该模型生成一个完成信息。
   3. 解析完成以提取操作，在我们的例子中，它应该是一个代码片段，因为我们正在使用 CodeAgent 。
   4. 动作执行。
   5. 将结果记录到 ActionStep 的内存中。
3. 在每个步骤结束时，如果agent包含任何函数调用（在 agent.step_callback 中），则会执行它们。   
### 实践时间！
以下我会展示两个示例，一个是huggingface 官方课程的例子，一个是自己设计的旅游助手。
#### 派对管家
Alfred要为Wayen家族筹办一场派对。需要做到以下几点：

- 选择派对上的音乐
- 为访客整理菜单
- 计算准备时间
- 在社区共享
- 使用 OpenTelemetry 和 Langfuse 📡 检查我们的派对管家

#### 旅游助手
### ToolCallingAgents

### Tools

## 让我们来实现一些Agents吧
### Retrieval Agents
### Multi-Agent System
### Vision and Browser agents
