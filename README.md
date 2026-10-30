# 企业知识问答 Agent（考试第 3 题）

**模式：真实 DeepSeek Tool Calling + 可离线复现的业务工具测试。**

帮助员工查询企业制度和产品事实。通过两种实际业务工具 `search_documents`（检索文档及候选段落 ID）和 `read_document`（支持可选 `paragraph_id` 的原文读取），模型可根据工具结果自主决定补查、回答、请求澄清或停止。回答采用原文 ID、段落 ID、版本日期和精确引文；冲突时不擅自选一项。所有文档均为**合成数据**，不代表任何真实公司的制度。

> **运行证据说明：** `examples/offline/` 是脚本化 Mock，只证明工程流程。`examples/live/` 保留历史 V2 的六组真实模型 Trace；`examples/live_v3/` 是 **V3 版本重新运行的六组真实 DeepSeek Trace（6/6 主要场景通过）**；`examples/live_mutation_v3/` 是正文变异 A/B 实测，包含一次连接失败及随后的成功重试。V3 本地单元测试 **45/45 通过**。请勿把 `.env` / API Key 提交到 GitHub。

## 1. 环境、安装（Python 3.10+）

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

将 `.env.example` 复制成 `.env`，填入你自己的 `DEEPSEEK_API_KEY`。默认模型是 DeepSeek 官方文档列出的 `deepseek-flash`，API 地址 `https://api.deepseek.com`。如你的账号支持其他模型，可修改 `DEEPSEEK_MODEL`。默认禁用思考模式，避免对 `reasoning_content` 的跨轮回传处理。

## 2. 评审快速入口

### 不需要 Key：自动化测试和离线示例

```bash
python -m unittest discover -s tests -v
python -m scripts.generate_offline_examples
python -m src.main --mode offline --case conflict --input examples/inputs/conflict.input.json
```

### 需要真实 DeepSeek Key：动态 Agent

```bash
python -m src.main --mode live --input examples/inputs/normal.input.json
python -m src.main --mode live --input examples/inputs/conflict.input.json
python -m src.main --mode live --input examples/inputs/follow_up.input.json

# 跑全部六个验收场景并保存真实输出和实际 Trace
python -m scripts.run_live_cases --out examples/live_v3
# 也可以先只验证两个
python -m scripts.run_live_cases --only normal conflict

# 不需要 Key：对照当前文档正文和被修改的副本
python -m scripts.run_live_mutation --scenario all --dry-run

# 需要 Key：真实模型文档替换实验，2 个场景共 4 次完整 Agent 运行
# D03 年假 3 -> 5 工作日；D02 报销 15 -> 30 自然日（冲突应解除）
python -m scripts.run_live_mutation --scenario all
```

`python -m scripts.run_live_cases` 将生成：

- `examples/live_v3/<case>.input.json`：新版本实际测试输入
- `examples/live_v3/<case>.output.json`：新版本实际模型回答、引用、任务状态
- `examples/live_v3/<case>.trace.json`：每次模型决策、工具参数、工具观测结果
- `examples/live_v3/manifest.json`：自动化初筛结果（仍须人工复核语义）

如果有 `heuristic_pass=False`，阅读实际 Trace 并修复问题后重新执行，不要将失败伪装成通过。

### 自定义问题

```bash
python -m src.main --mode live --question "星河协作是否支持离线编辑？"
python -m src.main --mode live --question "员工报销期限是多少天？" \
  --max-steps 8 --max-tool-calls 12 --output examples/my-check.json
```

`examples/inputs/follow_up.input.json` 展示可选对话历史格式：

```json
{
  "question": "那由谁来审批？",
  "history": [
    {"role": "user", "content": "年假要提前多久申请？"},
    {"role": "assistant", "content": "根据 D03 应至少提前3个工作日提交审批。"}
  ]
}
```

### 工具故障注入

```bash
python -m src.main --mode live \
  --input examples/inputs/tool_failure.input.json \
  --fail-tool search_documents:once
```

第 1 次 `search_documents` 返回结构化错误，**不会被代码自动重试**。模型观察错误后可以选择重试或结束，行为可在 Trace 中核查。也支持 `read_document:always`。

## 3. 项目结构

```text
src/
  main.py            CLI 参数、输入加载与运行入口
  agent.py           真实模型动态决策 -> 工具 -> 观察 -> 再决策
  provider.py        DeepSeek API 适配器与严格标识的 Offline Scripted Model
  tools.py           搜索、读取工具，输入校验，注入故障
  kb.py              可复现的检索、段落读取；期限事实来自当前文档原文
  evidence.py        审批/审批人/功能极性的有限原文一致性守卫
  prompts.py         Agent 业务指令与输出约定
  validation.py      引文溯源、数字+单位、主题相关性与正文冲突校验
  offline_replay.py  仅用于离线测试的固定模型动作
data/documents.json  六份合成制度与产品说明
tests/test_core.py               基础 Agent/工具/守卫测试
tests/test_mutation_and_evidence.py  正文变异、段落读取、语义反例测试
examples/inputs/     六类验收输入
examples/offline/    离线运行得到的实际输出及 Trace（**非真实 LLM**）
examples/live/       历史 V2 真实 DeepSeek 测试证据
examples/live_v3/    V3 六类验收的真实模型输出、Trace 和 manifest
examples/live_mutation_v3/  D03、D02 正文变异 A/B、连接失败及重试证据
scripts/             离线示例与真实模型验收
docs/                五份笔试要求的交付文档
```

## 4. 输出格式和状态

业务输出含 `status`、`answer`、`citations`、`conflicts`、`missing_information`、`follow_up_question`、`metrics`、`trace`，以及明确的 `mode`。

| 状态 | 含义 |
| --- | --- |
| `completed` | 已检索并有经验证的原文引用，且无已发现的未解决冲突 |
| `needs_clarification` | 文档冲突或需要用户澄清 |
| `insufficient_evidence` | 知识库缺少依据，不凭空回答 |
| `tool_error` | 模型调用失败或存在无法处理的工具错误 |
| `incomplete` | 引文不可信、执行上限、部分任务未完成等 |

Trace 不保存模型内部 Chain-of-Thought，只记录可观察的动作、参数、工具响应和最终状态。

## 5. 六类验收场景

| 案例 | 输入 | 关键验收 |
| --- | --- | --- |
| `normal` | 年假提前几天申请 | 有效文档原文与确定答案 |
| `conflict` | 报销期限多久 | D01 为30天、D02 为15天；展示双方并请求确认 |
| `out_of_scope` | 宠物托管补贴 | 无资料时拒绝编造 |
| `paraphrase` | 年假问题改写 | 相同事实与来源 |
| `follow_up` | 那由谁来审批 | 利用对话历史检索，引用 D03:p2 |
| `tool_failure` | 忘记密码怎么办 | 注入工具失败，观察实际调整或明确失败 |

另有单元测试验证工具参数、错误注入、轮次上限、未检索直接回答、非法引用、文档内容替换后的旧引用失效、跨段落证据合并、静态元数据陈旧问题，以及“真引文却得出错误结论”的部分反例。

## 6. 设计取舍、边界和安全

- 不依赖真实企业系统、不联网搜索；合成政策直接存放在 JSON 文件。
- 本地检索是简易关键词 / 字符 bigram 排序；支持给定数据集，但**不是生产级语义检索**。搜索返回 `matched_paragraph_ids`，`read_document(doc_id, paragraph_id?)` 可按段落读取；如果不传，则读取整份短文档。
- 真正的下一工具由 LLM 选择；工程代码仅处理参数、错误、限额、日志和引用安全保障。真实模式遇到无效 JSON/引用时，最多提供两次验证反馈让模型自行修正，不自动指定下一业务工具。
- `validation.py` 验证引用确实来自本轮 `read_document`、版本日期匹配、数字与单位一致；对明确问题增加引用主题相关性检查。报销期限冲突从**当前原文**抽取，不再依赖静态 `fact_value`。`evidence.py` 还能拒绝一部分明确反向的审批/审批人/支持功能说法，但**不能形式化证明任意自然语言语义蕴含**，仍须人工检查。
- 真实模型结果存在随机性；启用前必须检查实际通过率和文件，不保证六项首次都通过。
- 所有外部文档视为不可信数据。真实业务部署前还需权限、保密、审核、审计和更强检索。
- 模型 Key 仅放 `.env` 或环境变量；`.gitignore` 忽略 `.env`，不把真实员工数据放入仓库。

## 7. 最终交付检查

1. `python -m unittest discover -s tests -v` 全部通过。
2. 核对 `examples/live_v3/` 中已保存的六例真实 DeepSeek 输出与 Trace；如需重新复测，运行 `python -m scripts.run_live_cases --out examples/live_v3`。至少保留三组实际模型输入/输出与 Trace，并覆盖六种验收点。
3. 提交 `src/`、`tests/`、`data/`、`examples/inputs/`、`examples/offline/`、实际生成的 `examples/live/`、`docs/`、README、依赖和 `.env.example`。
4. 检查 `.env` 不受 Git 跟踪：`git status --short` / `git ls-files .env`；后者不应有输出。
5. 提交题号 **3**、GitHub URL 和最终完整 Commit SHA（`git rev-parse HEAD`）。如为私有仓库，授予指定评审账号访问权限。

## 8. 评审导航

- [需求与成功标准](docs/01-requirements.md)
- [架构、动态决策及技术取舍](docs/02-solution.md)
- [真实运行证据、V3 单测及文档变异实验](docs/03-evaluation.md)
- [AI 辅助及个人贡献记录](docs/04-ai-usage.md)
- [问题、学习与后续改进](docs/05-retrospective.md)

API 参考：[DeepSeek Tool Calls](https://api-docs.deepseek.com/guides/tool_calls/)；
[DeepSeek Models](https://api-docs.deepseek.com/quick_start/pricing/)；
[交付格式示例](https://github.com/lwz1357/agent-campus-submission-example)。

## 9. 审核用的真实问题与工程迭代证据

首次真实 DeepSeek 年假问答（V1）找到了 D03 正确原文，却因检索混入无关 D01/D02 报销制度、校验器把无关冲突当成当前问题的冲突而返回 `incomplete`。之后 V2 修复了检索过滤和冲突主题范围，用户实际复测得到 `completed`。前后同一输入对照：

| | 首次失败 | V2 修复后 |
| --- | ---: | ---: |
| 模型调用 | 7 | 3 |
| 工具调用 | 7 | 2 |
| 总 Token | 14,965 | 3,667 |

原有 `examples/live/` 的六组真实 DeepSeek 场景均来自 V2，并已保存历史结果与 Trace。

### V3：正文修改后行为必须跟着变

审核进一步发现：旧版校验依赖静态 `fact_value`，仅修改 D02 原文时可能出现过期冲突。本版：

1. 移除 `data/documents.json` 的 `fact_value`；使用 `conflict_group` 标识同主题，实际事实由 `src/kb.py` 从**最新段落原文**提取。
2. 当 D01=30 自然日、D02=15 自然日时，保持 `needs_clarification` 的冲突保护；当仅把 D02 正文改成 30 自然日时不应继续报告旧冲突。
3. 新增 `src/evidence.py` 的有限事实检查，覆盖审批必要性、错误审批人和产品功能支持极性。通过真实引用字符串仍不足以证明所有语义断言，不能将其宣传成通用 NLI。
4. `read_document` 增加可选 `paragraph_id`。多次读取不同段落的证据会合并，不会因后续局部读取而遗失已读依据。
5. 新增文档变异与语义反例等回归测试；当前版本 **45 项本地单元测试通过**。

**V3 实测已完成，证据已归档：**

- **45/45 本地单元测试通过**；另已完成 `--dry-run` 两种正文变异的程序检查。
- **六组真实 DeepSeek 回归通过**：22 次模型调用、19 次工具调用、34,766 Token；10 条引用均与本次 Trace 读取的原文一致。
- **D03 3→5 个工作日**：原始测试 `completed`；修改后的第一次 API 请求发生 `APIConnectionError`，结果为 `tool_error`；补测文件 `leave/modified_retry.json` 显示 `completed` 且正确引用新原文。
- **D02 15→30 自然日**：原始期限冲突返回 `needs_clarification`；修改后两份制度期限一致，真实模型返回 `completed`。
- `examples/live_mutation_v3/summary.json` 的 `all_pass=false` 是第一次 API 连接失败的真实历史记录，**不应篡改**；重试通过另见 `independent_review_after_retry.json` 和原始 `modified_retry.json`。
- 两组“文档替换实验”及六组 V3 完整回归结果已整理到 `docs/03-evaluation.md`。

需要进一步复现实验时：

```powershell
python -m unittest discover -s tests -v
python -m scripts.run_live_mutation --scenario all --dry-run

# 下列两条会真实调用 DeepSeek 并产生 API 费用
python -m scripts.run_live_mutation --scenario all
python -m scripts.run_live_cases --out examples/live_v3
```

**技术边界：** 审批人、审批要求、功能支持等事实的检查仍是有限规则，不是通用语义蕴含保证。针对较新制度只能根据原文判断替代关系，不能仅凭版本日期确定优先级。

### 交付前再确认

- 代码、`data/`、`tests/` 和五份 `docs/` 版本一致。
- 不提交 `.env`、Key、真实个人/员工信息；确认 `git ls-files .env` 无输出。
- 已保留 V3 真实测试与连接错误后的重试证据；以后若再次出现 `incomplete`，保留失败 Trace 并再迭代，不通过编辑 JSON 伪装成功。
- GitHub URL 和完整 Commit SHA 以最终提交为准。
