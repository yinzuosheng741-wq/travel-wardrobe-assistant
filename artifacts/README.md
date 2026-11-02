# 评测证据与本地输出

| 位置 | 用途 |
| --- | --- |
| `evaluation_manifest.json` | 当前离线质量门槛使用的报告指针 |
| `agent_evaluation_knowledge_20261030.json` | 指针绑定的基线，保留原路径和原始内容 |
| `evaluation.json`、`agent_evaluation.json`、`agent_evaluation_20261022_round2.json` | 源码快照兼容报告；属于历史记录，不作为最新效果声明 |
| `reports/agent/` | 历次 Agent 固定评测 |
| `reports/benchmarks/` | 扩展测评、对照与题集审计 |
| `reports/knowledge/` | 知识来源审核与结构检查 |
| `reports/rag/` | RAG 前后对照、来源健康、回答样本及关键点诊断 |
| `reports/releases/` | 已完成发布检查的证据清单 |
| `local/` | 本地验证、历史日志压缩归档与整理记录，Git 忽略 |
| `validation/` | `scripts/validate.py` 默认输出，Git 忽略；CI 单独上传，保留 14 天 |

最近洗护修复：[2026-11-02 契约重放](reports/rag/care-release-critical-20261102.json)为 23/23；[回答质量及独立性说明](../docs/reports/CARE_EVIDENCE_20261102.md)保留首次失败与后续回归。2026-10-31 的 [14/23 原始诊断](reports/rag/rag-critical-review-20261031.json)继续保留。外部独立人工审读仍待完成，24/24 回归不作为盲测准确率。

历史 JSON 的测量值、指纹及其内部原始路径不重写；移动只改变文件存放位置。复现实验应使用报告对应的代码、语料与题集版本。新实验另取文件名，不覆盖旧基线；仅将确认可公开的结果放入 `reports/`。密钥、数据库、用户照片与真实会话不得放入本目录的公开区域。
