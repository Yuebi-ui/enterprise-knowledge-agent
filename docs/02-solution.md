# 02 — 架构、动态决策与技术取舍

## 模块

1. `src/main.py`：CLI、输入格式、结果输出。
2. `src/provider.py`：DeepSeek OpenAI-compatible Tool Calling；独立 `ScriptedModel` 仅供离线复现。
3. `src/agent.py`：模型响应、工具执行、观察结果回传、再次请求模型，执行上限与 Trace。
4. `src/tools.py`：`search_documents` / `read_document`，JSON Schema、参数检查、错误注入。
5. `src/kb.py`：本地 JSON 检索、候选段落 ID、原文/版本读取；冲突仅使用主题分组，并从当前正文提取期限事实，不信任可过期的静态期限字段。
6. `src/validation.py`：验证实际读过的引文、日期、数字与单位、相关主题及源文本冲突，不代替模型下一步工具选择。
7. `src/evidence.py`：有限范围的源文本事实检查（审批必要性、审批人、功能支持极性），拒绝部分“引用真的，结论却反了”的情况。

## 真实动态决策流程

```text
User question + optional history
          |
   DeepSeek (tools=auto)
      /         \
 tool_calls    final JSON
    |              |
 verify args   verify citations/status
    |
 execute local business tool
    |
 append tool observation with tool_call_id
    |
 DeepSeek re-evaluates and decides next action
```

`tools=auto`，**不预先规定工具调用顺序**。模型可以因观察到完整/缺失/冲突/工具失败而产生不同工具调用或结束行为。

- 查询“年假提前多久”：预期搜索 D03、读取原文后给出带引用答案。
- 查询“报销期限多久”：预期搜索 D01 和 D02、分别读取原文、发现优先级不明、向用户澄清。
- 注入 `search_documents:once`：工具返回结构化错误；代码不自动恢复，下一步由模型自主决定。
- 追问“那由谁审批”：模型应根据上下文主动补足搜索主题。

`src/agent.py` 的 `model_decision` 仅记录工具名、结构化参数、工具观测和模型最终答案；不要求或记录私有 Chain-of-Thought。

## 状态机

- `completed`：完成；前提是已经实际检索、读取、使用有效引用且无已知未解决冲突。
- `needs_clarification`：查到信息不一致或缺乏适用条件。
- `insufficient_evidence`：未找到足够证据，无依据不回答。
- `tool_error`：模型/工具不可用（错误应可观察）。
- `incomplete`：达到最大步骤/工具调用数、格式错误、引文非法或核对不全。

## 引用契约

每项证据包含 `doc_id`、`paragraph_id`、`version_date`、`quote`。
验证器拒绝不存在的段落、未读文件、不同版本日期、与原文不匹配的 quote；对于从当前正文明确检出的 D01/D02 冲突，必须读取并引用双方，且不能把唯一期限标为确定答案。若把 D02 原文改为与 D01 相同的 30 个自然日，则不再沿用旧冲突标签。

`read_document` 可选 `paragraph_id`，精确段落模式与默认整份短文档模式兼容。多轮读取同一文档不同段落时，会合并已读证据，避免覆写导致已读引文失效。

`src/evidence.py` 另外检测显式的审批要求反转、审批人不匹配、产品支持/不支持反转。强词面匹配下也校验引用与当前问题的主题相关性。这些都是**面向当前合成语料、可执行且可单测的有限守卫**，不是全语义 NLI；不能把它宣传成对任意语义陈述都正确。复杂事实仍需人工审阅或后续独立语义校验。

## 框架取舍

考虑考试建议 3 小时、要求 CLI/HTTP/Notebook 任一即可，选用原生 Tool Calling + 小型确定性 Controller，不增加多 Agent、Redis、Chroma、LangGraph 基础设施。该实现沿用先前 Deep Research 项目的检索—证据—验证设计思路，但不复制其部署复杂度。考核说明明确：框架名称本身不加分。

## 限制及故障边界

- 本地检索为轻量关键词/bigram 排序，不支持大规模语义搜索；可以替换为向量索引或混合检索。
- 真实 DeepSeek 的工具选择可能存在变异；应查看真实 Trace，不得把 mock 固定路径当作模型能力。
- 禁用 thinking 模式以保证 SDK Tool Calling 格式易于复现；非思考模式也可以通过多轮工具结果完成 ReAct。
- 达到上限直接返回 incomplete；无工具层无限自动重试。
- 所有 API 请求由考生本地使用自己的密钥完成，模型故障和密钥缺失不能伪装为成功。

## 文档替换实验设计（新增）

测试分两层，且**不能混淆离线证据与真实模型证据**。

第一层：无需 API 的确定性正文变异实验：

```bash
python -m scripts.run_live_mutation --scenario all --dry-run
python -m unittest discover -s tests -v
```

它把变异数据保存在 `examples/live_mutation_v3/`，验证：

- **D03**：年假提前期限原文从 3 工作日改为 5 工作日，旧引文不应通过。
- **D02**：报销期限原文从 15 自然日改为 30 自然日，D01/D02 不再构成该期限的事实冲突。只有变动的数据副本受影响；原始 `data/documents.json` 保持不变。
- **反例**：真实 D03 引文 + “不需要审批”、真实 D03:p2 + 错误审批人、D06:p2 + “支持离线编辑”均被新校验拦截。

第二层：有 Key 后的**真实 DeepSeek A/B 对照**：

```bash
python -m scripts.run_live_mutation --scenario all
```

每个场景分别对原始和变异数据完整调用一次 Agent（两个场景共四次完整 Agent 运行），保存原文副本、实际输出及 Trace。预期 D03 的答案值变化，以及 D01/D02 的最终状态由 `needs_clarification` 变为 `completed`。**模型动作与最终状态的变化必须看真实 Trace；不能把离线条件判断当成模型决策。**

> 限制：当前冲突事实抽取针对 `reimbursement_deadline` 合成语料；对于无法可靠解析的其他制度主题，不做无依据的“无冲突”保证。检索仍为小语料 lexical ranking，语义召回和完整语义蕴含不在本版能力范围。
