# 知识库

> 本文档由 AI 自动翻译。如有任何不准确之处，请参考 [英文原版](/en/cloud/use-dify/knowledge/readme)。

## 简介

在 Dify 中，你可以将自有数据作为「知识」集成到 AI 应用中。通过为大语言模型（LLM）提供特定领域的上下文信息，知识能够让 LLM 的回复更加准确、相关，并显著减少幻觉。

这得益于检索增强生成（RAG）技术。其核心在于：LLM 不再只依赖预训练的公开数据，还会将你的自定义知识作为额外的事实来源：

1. （检索）处理用户提问时，系统会先从已集成的知识库中 **检索最相关的信息**。
2. （增强）检索到的信息会与用户的原始问题打包，作为 **增强的上下文** 发送给 LLM。
3. （生成）LLM 基于这些上下文 **生成更精准的答案**。

知识存储在知识库中。你可以创建多个知识库，分别适配不同领域、场景或数据源，并按需集成到应用中。

## 应用场景

借助 Dify 知识库，你可以打造基于自有数据和特定领域知识的 AI 应用。常见场景包括：

* **智能客服机器人**：让问答机器人基于最新的产品文档、FAQ 和故障排查指南，智能回复客户问题。
* **企业内部知识门户**：为员工构建 AI 搜索与问答系统，快速查询公司政策与流程。
* **内容生成工具**：根据特定背景资料，智能生成报告、文章或邮件。
* **科研与分析应用**：检索和总结学术论文、市场报告、法律文档等专业知识，辅助研究与分析。

## 创建知识库

* **[创建即用型知识库](/zh/cloud/use-dify/knowledge/create-knowledge/introduction)**：导入数据，设置处理规则，其余一切交给 Dify。简单高效，新手友好。
* **[构建自定义知识库](/zh/cloud/use-dify/knowledge/knowledge-pipeline/readme)**：自定义步骤和集成，编排更复杂、灵活的数据处理流程。
* **[连接外部知识库](/zh/cloud/use-dify/knowledge/connect-external-knowledge-base)**：通过 API 直接同步外部知识库，无需迁移即可利用现有数据。

## 管理与优化知识库

* **[维护知识库内容](/zh/cloud/use-dify/knowledge/manage-knowledge/maintain-knowledge-documents)**：对文档及其分段进行查看、添加、修改和删除等操作，使知识库内容保持最新、准确、相关。
* **[测试召回效果](/zh/cloud/use-dify/knowledge/test-retrieval)**：模拟用户提问，测试知识库召回效果。
* **[利用元数据增强检索](/zh/cloud/use-dify/knowledge/metadata)**：为文档添加元数据，实现基于文档筛选的检索，进一步提升检索精度。
* **[调整知识库设置](/zh/cloud/use-dify/knowledge/manage-knowledge/introduction)**：随时调整索引方式、嵌入模型和检索策略等设置。

## 使用知识库

**[集成到应用](/zh/cloud/use-dify/knowledge/integrate-knowledge-within-application)**：将自定义知识集成到你的 AI 应用中。

***
> ## Documentation Index
> Fetch the complete documentation index at: https://docs.dify.ai/llms.txt
> Use this file to discover all available pages before exploring further.

# Workflow 与 Chatflow

> 构建将 AI 模型、工具和逻辑组合为可靠、可重复流程的 Agentic 工作流

> 本文档由 AI 自动翻译。如有任何不准确之处，请参考 [英文原版](/en/cloud/use-dify/build/workflow-chatflow)。

## 为什么需要 Agentic 工作流

AI 模型功能强大，但单独使用时可能不够可预测，它们可能会产生幻觉、遗漏步骤或生成不一致的输出。在生产环境中，尤其是对可靠性有要求的团队和企业，你需要对 AI 的运行方式有更多控制。

Agentic 工作流通过将 AI 能力嵌入结构化、可重复的流程来解决这一问题。与其依赖单个模型自行解决所有问题，不如设计一个流程来逐步编排模型、工具和逻辑，包含明确的条件、检查点和回退路径。

AI 仍然承担繁重的工作，但在你定义的边界内运行。



## Workflow vs. Chatflow

Dify 提供两种应用类型来构建 Agentic 工作流：**Workflow** 和 **Chatflow**。两者都基于共享的可视化画布和节点系统构建。

要构建流程，仅需连接各个节点。每个节点处理一个特定步骤，如调用模型、检索知识、运行代码或基于条件分支。大部分工作是 **拖拽、连接和配置**，只有当现有节点无法满足特定逻辑时，才需要考虑编写代码。

两者的核心区别在于用户如何与应用交互：

* **工作流** 从头到尾运行一次。

  它接收输入，通过流程处理，然后返回结果。适用于自动报告生成、数据处理管道或批处理等任务。

* **Chatflow** 增加了对话层。

  用户通过聊天界面进行交互，每条消息在生成回复前，都会触发你设计的流程。适用于交互式助手、引导式问答，或任何需要在每次回复背后进行结构化处理的对话场景。

  <Tip>
    Chatflow 支持内容审核、文字转语音等可选功能。详见 [应用工具箱](/zh/cloud/use-dify/build/additional-features)。
  </Tip>

两者的起始和结束节点也不同：

|      | Workflow                                                                                       | Chatflow                                    |
| :--- | :--------------------------------------------------------------------------------------------- | :------------------------------------------ |
| 起始节点 | [用户输入](/zh/cloud/use-dify/nodes/user-input) 或 [触发器](/zh/cloud/use-dify/nodes/trigger/overview) | 用户输入                                        |
| 结束节点 | [输出](/zh/cloud/use-dify/nodes/output)（可选）                                                      | [直接回复](/zh/cloud/use-dify/nodes/answer)（必需） |

触发器可按计划、Webhook 或集成事件自动运行工作流。Chatflow 始终由用户消息启动，因此没有触发器。

关于两种应用的编排方式，详见 [编排逻辑](/zh/cloud/use-dify/build/orchestrate-node)。





# 编排逻辑

> 构建 Workflow 或 Chatflow 时，如何排列、嵌套与复用节点

> 本文档由 AI 自动翻译。如有任何不准确之处，请参考 [英文原版](/en/cloud/use-dify/build/orchestrate-node)。

## 串行与并行执行

<Frame>
  <img src="https://mintcdn.com/dify-6c0370d8/xGr6gOhRwlVIl2gB/images/use-dify/workflow/serial-vs-parallel-execution.png?fit=max&auto=format&n=xGr6gOhRwlVIl2gB&q=85&s=c8d8b6a82092072fb8dac32973794488" alt="串行与并行执行" width="1896" height="568" data-path="images/use-dify/workflow/serial-vs-parallel-execution.png" />
</Frame>

构建工作流时，可将节点按串行或并行方式排列：

* **串行排列时**，节点依次运行。每个节点可读取链中任何前序节点的变量。

* **并行排列时**，节点同时运行，彼此无法读取变量；但当并行分支汇合后，下游节点可读取所有分支的变量。

<Info>
  单条执行路径最多支持 50 个节点。
</Info>

## 节点复用

除用户输入节点外，所有节点都可在同一工作流内、跨工作流或跨 Dify 实例复制粘贴，但不同 Dify 版本之间可能存在兼容性问题。

<Note>
  跨工作流或跨 Dify 实例粘贴时，Dify 页面需通过 HTTPS 加载，或通过环回地址（如 `http://localhost` 或 `http://127.0.0.1`）访问。
</Note>

粘贴节点时，节点配置随之迁移；但依赖周边环境的内容会在目标位置重新评估：

* **工作流资源**，如变量
* **工作空间资源**，如集成和知识库

## 迭代与循环

若某些节点需要多次运行（按列表每项运行一次，或直到满足条件才停止），可将其置于 [迭代](/zh/cloud/use-dify/nodes/iteration) 或 [循环](/zh/cloud/use-dify/nodes/loop) 节点内部。





# Agent

> 构建一次 Agent，既可作为独立应用使用，也可嵌入工作流

> 本文档由 AI 自动翻译。如有任何不准确之处，请参考 [英文原版](/en/cloud/use-dify/build/new-agent/overview)。

<Info>
  新 Agent 目前处于 Beta 阶段。
</Info>

Agent 是一名 AI 员工：设置一次，之后随时给它派活。它与 [旧版 Agent 应用](/zh/cloud/use-dify/build/agent) 是不同类型的 Agent：

* 它在 **自己的沙箱** 中工作：可运行命令、安装程序、读写文件，因此能承担开放式的工作，而不只是调用你配置好的工具。

* 你只需 [构建](/zh/cloud/use-dify/build/new-agent/build) **一次**，就能以 **两种方式** 使用：单独作为聊天应用，或作为工作流中的 [一步](/zh/cloud/use-dify/nodes/agent#new-agent)。

Agent 的创建、配置和管理都在 **Agents** 页面进行。可随时打开任一 Agent 继续打磨，或回看早期版本。

## 能力与任务

Agent 把「它是什么」和「你要它做什么」分开：

* **能力（可视为 Agent 的灵魂）决定了 Agent 是谁。**

  包括你编写的角色和提示词、它运行的模型，以及你提供的 Skill、Dify 工具和文件。塑造一次后，可随着你逐渐摸清 Agent 需要什么而不断打磨。

* **任务是你在一次运行中要它做的事。**

  Agent 独立工作时，任务是你发给它的消息；在工作流中时，任务是你给节点的指令。

这和招聘是一回事：先按能力选人，再交给对方具体任务。好结果两者缺一不可：人要选对，任务也要交代清楚。

## Agent 的两种用法

### 独立使用

Agent 作为独立的聊天应用运行。你通过聊天给它派任务，还可将它发布为 Web 应用，或通过服务 API 调用。

每场对话都有自己的记忆：Agent 能记住多少取决于模型的上下文窗口，新开对话则从零开始。

当一名能干的员工足以独立达成目标时，选这种方式：比如查资料后作答的客服助手，或收集来源并总结要点的调研 Agent。

### 在工作流中使用

将 Agent 请进 [Agent 节点](/zh/cloud/use-dify/nodes/agent#new-agent)，负责整个流程中的一步。在节点上只需设置要完成的任务。这就像请同事帮忙办一件事：你说明任务，对方带着自己的本事来完成。

当工作需要流程结构时，选这种方式：多个步骤按固定顺序执行、按条件分支、用到其他类型的节点，或几个专精不同方向的 Agent 相互接力。

无论以哪种方式运行，Agent 的会话都汇总在同一处：**日志** 页面，每一行都标明来源，即该 Agent 的 Web 应用或使用它的工作流。

