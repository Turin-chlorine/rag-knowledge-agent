# 中文 RAG 评测轨道

这套工具把公开语料放在 `data/benchmarks/`，与网页知识库分开。准备数据时固定随机种子、上游版本和文件 SHA-256；问答标签不会写进待检索文档。生成的语料、索引、回答与评审文件都位于已被 `.gitignore` 排除的 `data/` 下。

## 三个独立轨道

| 轨道 | 上游 | 本项目中的用途 | 标签限制 |
| --- | --- | --- | --- |
| CRUD-RAG | [官方仓库](https://github.com/IAAR-Shanghai/CRUD_RAG) | 400 篇中文新闻文档、120 道来源相关问答与 20 道近似无答案候选；测完整 RAG。背景文档一半按词面相近选，一半固定随机抽样 | `news1` 等字段给出来源文档，但参考答案经改写；片段证据与无答案边界需人工核实 |
| MIRACL-zh | [官方仓库](https://github.com/project-miracl/miracl) | 一个中文语料分片中抽 1200 个文段及 50 道有正例的查询；测检索。额外文段含词面相近的未标注干扰项 | 查询来自上游 dev 集；使用官方 qrels，未标注文段不当作负例；本地子集不能对照完整基准分数 |
| RGB-zh | [官方仓库](https://github.com/chen700564/RGB) | 60 个给定片段的噪声、整合和矛盾压力案例；测生成 | 上游片段多为搜索摘要，不是完整可入库文档；许可限非商业用途 |

CRUD-RAG 仓库没有声明仓库级许可，且原文来自新闻媒体；研究测试前仍须核对原文再分发权限。MIRACL 数据卡标记 Apache-2.0，但正文源于维基百科，应保留来源说明。RGB 仓库明确说明采用 CC BY-NC-SA 4.0 且仅限非商业使用。工具不会把公开原文提交到 Git。

CRUD 来源文档直接取自上游问答记录的 `news1`–`news3` 字段，在当前固定样本中长度为 206–960 字；它们是可复现的上游文本，不能据此认定已收集到新闻网站的完整文章正文。

## 生成与运行

在项目根目录运行，使用已经安装 `requirements.txt` 的 Python 环境：

```powershell
venv\Scripts\python.exe benchmarks\prepare.py crud
venv\Scripts\python.exe benchmarks\prepare.py miracl --questions 50 --doc-count 1200
venv\Scripts\python.exe benchmarks\prepare.py rgb --questions-per-type 20

venv\Scripts\python.exe benchmarks\evaluate.py index crud
venv\Scripts\python.exe benchmarks\evaluate.py index miracl
venv\Scripts\python.exe benchmarks\evaluate.py run crud --split test --mode retrieval
venv\Scripts\python.exe benchmarks\evaluate.py run miracl --mode retrieval
```

`index` 复用项目的切片和中文 Embedding，`run` 复用实际两阶段检索；索引在各自轨道目录中，不覆盖 `data/vector_store.json`。MIRACL 首次准备需下载约 100 MB 的一个中文语料分片；可通过 `--corpus <路径>` 使用已有官方 JSONL 或 JSONL.GZ。首次加载 Embedding 和重排模型也可能联网。

完整回答测试会调用配置在 `.env` 中的 DeepSeek API，并产生调用费用。先用 `--limit 3` 验证，再运行全量；限制运行的结果文件名含 `_limit3`，不能作为完整评测成绩。端到端结果文件名还包含 UTC 时间，逐题保存已完成的付费回答。

```powershell
venv\Scripts\python.exe benchmarks\evaluate.py run crud --split test --mode e2e --limit 3
venv\Scripts\python.exe benchmarks\evaluate.py run rgb --mode e2e --limit 3
```

`data/benchmarks/<轨道>/prepared/manifest.json` 记录样本规模、随机种子、上游修订和源文件哈希。`documents.jsonl` 保留来源元数据；实际入库文档在 `documents/`；`cases.jsonl` 才包含题目与参考答案。
每次检索运行还会生成 `*_summary.json` 和便于阅读的 `*_report.md`；已有结果可用 `evaluate.py report --results <结果文件>` 重新生成报告，不重复调用模型。召回的 @20/@8 分别以实际粗排前 20 个切片和重排前 8 个切片为范围，再按切片所属来源文档计算。

## 人工核实与指标口径

1. 复制 CRUD 的 `gold_review_template.jsonl`，对每道可回答题，在 `gold_doc_ids` 所指的原文中确认答案事实与具体片段。`evidence_candidates` 只是自动给出的候选原文句，不能直接视为真值。对 20 道近似无答案题，先看 `near_no_answer_audit.jsonl` 列出的 15 篇最相近文档及全文精确答案检查，再确认**全部 400 篇入库文档**都不支持答案。未核实的题不计入最终幻觉率或拒答率。
2. 跑 `--mode e2e` 后，使用 `evaluate.py merge-gold --results <本次运行结果> --reviews <填好的 gold_review 文件>` 合并金标准核实结果，再逐题填写 `answer_correct`、`action_correct`、`citation_correct`、`supported_claims`、`unsupported_claims`。可回答的 CRUD 题还须在 `verified_evidence` 中为每个来源文档填写 `{"doc_id":"...","quote":"原文中的逐字片段"}`；评分器会核对片段确实在入库原文中。把回答拆成可单独核对的事实陈述；每条无原文支持的陈述记为一个 `unsupported_claim`。确认引用的编号确实指向支持该陈述的原文。
3. 评分时只纳入核实完成的题；未核实项和其数量明确显示。`unsupported_claim_rate` 是**无依据事实陈述数 / 全部已评审事实陈述数**；`answer_with_unsupported_claim_rate` 是含无依据事实的回答占有事实陈述回答的比例。它们与旧 110 题的二元幻觉标签不同。正确拒答单独看 `action_accuracy`，不能用回答格式字符串替代人工判断。

```powershell
venv\Scripts\python.exe benchmarks\evaluate.py score --results <本次运行的结果.jsonl> --reviews <填好的人工评审.jsonl>
```

评分会生成 `*_scored.json` 和 `*_scored_report.md`。尚无人复核时，报告会显示已评审 0 题，各回答指标为「待人工核实」，不会把空值写成 0%。

报告按题型列出样本数、检索阶段来源命中率和人工核实后的回答指标。只用 `dev` 调整阈值、Prompt 或模型；`test` 固定用于最后检验。CRUD 的来源文档命中率在人工证据确认前只是诊断值。MIRACL 的子集指标不能当作论文基准分数。

## 飞书 200 题评测

飞书评测表记录了项目 API 在 `temperature=0.3` 下的 200 题运行与逐题判定。题目取自 CRUD-RAG 中文新闻问答数据的固定子集，包含单文档、双文档、三文档各 60 题，以及近邻无答案题 20 题。题目、参考答案与待检索原文分开存放，避免答案泄漏。

所有 200 题均有评分与判定结果。当前汇总指标为：总体回答正确率 187/200（93.50%），幻觉率 18/200（9.00%）；可回答的 180 题来源文档召回@8 为 353/360（98.06%），全部预期来源命中题率为 174/180（96.67%）。20 道无答案题没有正例来源文档，不计入来源召回分母。幻觉率尚高于产品目标 5%。

复现工具包括 `build_upload200.py`、`build_upload200_v2.py`、`run_upload200.py` 和 `publish_upload200_v2.py`；结果文件及逐题审计保存在被 Git 忽略的 `data/benchmarks/` 下。记录文件名用于追溯构建与运行过程，汇总指标统一按 200 题报告。该固定子集分数不代表 CRUD-RAG 完整基准成绩。
